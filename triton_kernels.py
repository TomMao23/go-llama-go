"""Triton kernels for accelerating the Llama model in llama.py.

All optimizations of the model come from the fused kernels defined here:

  - rms_norm:        fused RMSNorm (single kernel instead of pow/mean/rsqrt/mul/mul)
  - add_rms_norm:    fused residual add + RMSNorm
  - rope:            fused RoPE application + layout permute ([B,S,H,D] -> [B,H,S,D])
  - swiglu:          fused SiLU(gate) * up, written in place over the gate half
  - flash_attn:      fused RoPE + causal flash attention with native GQA support
                     (no materialized S x S score matrix, no KV repeat_interleave)
  - argmax:          fused row argmax (first-max-wins, like torch.argmax)

Kernel output buffers come from a grow-only arena to avoid per-call
torch.empty dispatch cost; every buffer is consumed before its slot is
reused, which the model's sequential dataflow guarantees.
"""

import math

import torch
import triton
import triton.language as tl

_buffers = {}


def _buf(key, shape, dtype, device):
    numel = 1
    for s in shape:
        numel *= s
    buf = _buffers.get(key)
    if buf is None or buf.numel() < numel or buf.dtype != dtype or buf.device != device:
        buf = torch.empty(numel, dtype=dtype, device=device)
        _buffers[key] = buf
    return buf[:numel].view(shape)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
@triton.jit
def _rms_norm_kernel(
    X, W, Y,
    stride_x, stride_y,
    N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # Emulate the baseline's bf16 rounding chain: pow -> mean -> +eps -> rsqrt
    # each round back to the storage dtype (reduction accumulates in fp32).
    xd = x.to(X.dtype.element_ty)
    var = tl.sum((xd * xd).to(tl.float32), axis=0) / N
    var = var.to(X.dtype.element_ty).to(tl.float32)
    rrms = tl.rsqrt(var + EPS)
    rrms = rrms.to(X.dtype.element_ty).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    tl.store(Y + row * stride_y + cols, (xd * rrms.to(X.dtype.element_ty) * w.to(X.dtype.element_ty)).to(Y.dtype.element_ty), mask=mask)


def rms_norm(x, weight, eps):
    shape = x.shape
    x_2d = x.reshape(-1, shape[-1])
    y = _buf("rms", x_2d.shape, x.dtype, x.device)
    rows, n = x_2d.shape
    _rms_norm_kernel[(rows,)](
        x_2d, weight, y,
        x_2d.stride(0), y.stride(0),
        N=n, EPS=eps, BLOCK=triton.next_power_of_2(n),
        num_warps=8,
    )
    return y.view(shape)


# ---------------------------------------------------------------------------
# Fused residual add + RMSNorm:
#   new_residual = residual + x
#   out          = rms_norm(new_residual)
# ---------------------------------------------------------------------------
@triton.jit
def _add_rms_norm_kernel(
    X, RES, W, NEW_RES, Y,
    stride_x, stride_res, stride_y,
    N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(RES + row * stride_res + cols, mask=mask, other=0.0).to(tl.float32)
    # Round the residual sum back to the storage dtype, mirroring the
    # semantics of the in-place bf16 `hidden_states += ...` in the baseline.
    new_res = (x + res).to(NEW_RES.dtype.element_ty)
    tl.store(NEW_RES + row * stride_y + cols, new_res, mask=mask)

    t = new_res.to(tl.float32)
    td = new_res
    var = tl.sum((td * td).to(tl.float32), axis=0) / N
    var = var.to(NEW_RES.dtype.element_ty).to(tl.float32)
    rrms = tl.rsqrt(var + EPS)
    rrms = rrms.to(NEW_RES.dtype.element_ty).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    tl.store(Y + row * stride_y + cols, (td * rrms.to(NEW_RES.dtype.element_ty) * w.to(NEW_RES.dtype.element_ty)).to(Y.dtype.element_ty), mask=mask)


def add_rms_norm(residual, x, weight, eps):
    shape = residual.shape
    res_2d = residual.reshape(-1, shape[-1])
    x_2d = x.reshape(-1, shape[-1])
    new_res = _buf("add_res", res_2d.shape, residual.dtype, residual.device)
    y = _buf("add_norm", res_2d.shape, residual.dtype, residual.device)
    rows, n = res_2d.shape
    _add_rms_norm_kernel[(rows,)](
        x_2d, res_2d, weight, new_res, y,
        x_2d.stride(0), res_2d.stride(0), y.stride(0),
        N=n, EPS=eps, BLOCK=triton.next_power_of_2(n),
        num_warps=8,
    )
    return new_res.view(shape), y.view(shape)


# ---------------------------------------------------------------------------
# RoPE: input [B, S, H, D] -> output [B, H, S, D] (permute fused into kernel).
# Handles q and k in a single launch. theta (per-dim frequencies, rounded to
# the storage dtype exactly like the baseline tables) is passed in; angle and
# sin/cos are computed inline to avoid materializing [S, HALF] tables.
# ---------------------------------------------------------------------------
@triton.jit
def _rope_kernel(
    Q, K, THETA, QO, KO,
    stride_qt, stride_kt,
    H_Q: tl.constexpr, H_KV: tl.constexpr,
    S, D: tl.constexpr, HALF: tl.constexpr,
):
    token = tl.program_id(0)
    h = tl.program_id(1)
    b = token // S
    s = token % S

    d = tl.arange(0, HALF)

    # angle = pos * theta computed in the storage dtype, then sin/cos rounded
    # back to it, mirroring the baseline's bf16 sin/cos tables.
    theta = tl.load(THETA + d)
    angle = s.to(THETA.dtype.element_ty) * theta
    sin = tl.sin(angle.to(tl.float32)).to(THETA.dtype.element_ty)
    cos = tl.cos(angle.to(tl.float32)).to(THETA.dtype.element_ty)

    if h < H_Q:
        x_base = Q + token * stride_qt + h * D
        o_base = QO + (b * H_Q + h) * (S * D) + s * D
    else:
        hk = h - H_Q
        x_base = K + token * stride_kt + hk * D
        o_base = KO + (b * H_KV + hk) * (S * D) + s * D

    x0 = tl.load(x_base + d)
    x1 = tl.load(x_base + HALF + d)

    tl.store(o_base + d, x0 * cos - x1 * sin)
    tl.store(o_base + HALF + d, x0 * sin + x1 * cos)


def rope(q, k, theta, num_q_heads, num_kv_heads):
    """q: [B, S, Hq*D], k: [B, S, Hkv*D] views (last dim contiguous, may be
    slices of one fused qkv tensor), theta: [D/2] RoPE frequencies in the
    model dtype. Returns rotated (q, k) in contiguous [B, H, S, D] layout.
    """
    b, s = q.shape[0], q.shape[1]
    d = q.shape[-1] // num_q_heads
    assert q.shape[-1] % num_q_heads == 0 and k.shape[-1] == num_kv_heads * d
    q_out = torch.empty((b, num_q_heads, s, d), dtype=q.dtype, device=q.device)
    k_out = torch.empty((b, num_kv_heads, s, d), dtype=k.dtype, device=k.device)
    _rope_kernel[(b * s, num_q_heads + num_kv_heads)](
        q, k, theta, q_out, k_out,
        q.stride(1), k.stride(1),
        H_Q=num_q_heads, H_KV=num_kv_heads, S=s, D=d, HALF=d // 2,
        num_warps=4,
    )
    return q_out, k_out


# ---------------------------------------------------------------------------
# SwiGLU: silu(gate) * up, reading both halves from one fused [*, 2I] tensor
# produced by a single gate+up GEMM. The result is written in place over the
# gate half, so no output allocation is needed.
# ---------------------------------------------------------------------------
@triton.jit
def _swiglu_kernel(GU, TOTAL, HALF: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    row = offs // HALF
    col = offs % HALF

    g = tl.load(GU + row * (2 * HALF) + col, mask=mask, other=0.0)
    u = tl.load(GU + row * (2 * HALF) + HALF + col, mask=mask, other=0.0)
    # Emulate baseline rounding: silu result rounds to storage dtype before
    # the elementwise multiply.
    silu = (g.to(tl.float32) * tl.sigmoid(g.to(tl.float32))).to(GU.dtype.element_ty)
    y = silu * u

    tl.store(GU + row * (2 * HALF) + col, y.to(GU.dtype.element_ty), mask=mask)


def swiglu(gate_up):
    """gate_up: [..., 2 * I] with gate in the first half, up in the second.
    Computes in place and returns a view over the first half."""
    assert gate_up.is_contiguous()
    half = gate_up.shape[-1] // 2
    total = gate_up.numel() // 2
    block = 1024
    grid = (triton.cdiv(total, block),)
    _swiglu_kernel[grid](gate_up, total, HALF=half, BLOCK=block, num_warps=4)
    return gate_up[..., :half]


# ---------------------------------------------------------------------------
# Flash attention with causal mask, native GQA and fused RoPE.
#   q: [B, S, Hq*D] strided view (slice of the fused qkv GEMM output)
#   k: [B, S, Hkv*D] strided view, theta: [D/2] RoPE frequencies
#   v: [B, Hkv, S, D] (arbitrary strides; may be a strided view)
#   output: [B, S, Hq * D] (arena buffer, ready for o_proj)
# RoPE is applied in-kernel: QK^T = Q0r@K0r^T + Q1r@K1r^T over the two
# half-dim tiles, which equals the dot of fully rotated q and k while
# avoiding a separate rope launch and its buffers.
# ---------------------------------------------------------------------------
@triton.jit
def _flash_attn_kernel(
    Q, K, V, THETA, O,
    stride_qb, stride_qm,
    stride_kb, stride_km,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_om,
    H_Q: tl.constexpr, H_KV: tl.constexpr, GROUP: tl.constexpr,
    S, SCALE: tl.constexpr,
    D: tl.constexpr, HALF: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H_Q
    h = pid_bh % H_Q
    h_kv = h // GROUP

    dt = THETA.dtype.element_ty

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    d_half = tl.arange(0, HALF)
    d_full = tl.arange(0, D)
    m_mask = offs_m < S

    theta = tl.load(THETA + d_half)

    # Rotate the q rows of this block (positions offs_m), matching the
    # baseline's bf16 table arithmetic step by step.
    angle_m = offs_m.to(dt)[:, None] * theta[None, :]
    sin_m = tl.sin(angle_m.to(tl.float32)).to(dt)
    cos_m = tl.cos(angle_m.to(tl.float32)).to(dt)

    q_base = Q + b * stride_qb + h * D
    q0 = tl.load(
        q_base + offs_m[:, None] * stride_qm + d_half[None, :],
        mask=m_mask[:, None], other=0.0,
    )
    q1 = tl.load(
        q_base + offs_m[:, None] * stride_qm + HALF + d_half[None, :],
        mask=m_mask[:, None], other=0.0,
    )
    q0r = q0 * cos_m - q1 * sin_m
    q1r = q0 * sin_m + q1 * cos_m

    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    k_base = K + b * stride_kb + h_kv * D
    v_base = V + b * stride_vb + h_kv * stride_vh

    hi = tl.minimum(S, (pid_m + 1) * BLOCK_M)
    for n0 in range(0, hi, BLOCK_N):
        n_offs = n0 + offs_n
        n_mask = n_offs < S

        # Rotate the k rows of this block (positions n_offs).
        angle_n = n_offs.to(dt)[:, None] * theta[None, :]
        sin_n = tl.sin(angle_n.to(tl.float32)).to(dt)
        cos_n = tl.cos(angle_n.to(tl.float32)).to(dt)

        k0 = tl.load(
            k_base + n_offs[:, None] * stride_km + d_half[None, :],
            mask=n_mask[:, None], other=0.0,
        )
        k1 = tl.load(
            k_base + n_offs[:, None] * stride_km + HALF + d_half[None, :],
            mask=n_mask[:, None], other=0.0,
        )
        k0r = k0 * cos_n - k1 * sin_n
        k1r = k0 * sin_n + k1 * cos_n

        # Emulate the baseline score rounding chain: matmul output rounds to
        # the storage dtype, then the scale multiply rounds again.
        qk = tl.dot(q0r, tl.trans(k0r)) + tl.dot(q1r, tl.trans(k1r))
        qk = qk.to(dt)
        qk = (qk.to(tl.float32) * SCALE).to(dt)
        causal = offs_m[:, None] >= n_offs[None, :]
        qk = tl.where(causal & n_mask[None, :], qk, float("-inf"))
        # Single-block softmax (BLOCK_N >= S at launch): no online rescaling,
        # so the math is exactly max -> exp -> sum -> divide, like the
        # baseline's torch.softmax (fp32 opmath, probs rounded once).
        m = tl.max(qk.to(tl.float32), axis=1)
        p = tl.exp(qk.to(tl.float32) - m[:, None]).to(dt)
        l = tl.sum(p.to(tl.float32), axis=1)
        probs = (p.to(tl.float32) / l[:, None]).to(dt)

        v = tl.load(
            v_base + n_offs[:, None] * stride_vn + d_full[None, :],
            mask=n_mask[:, None], other=0.0,
        )
        acc = tl.dot(probs, v)

    tl.store(
        O + b * stride_ob + offs_m[:, None] * stride_om + h * D + d_full[None, :],
        acc.to(O.dtype.element_ty),
        mask=m_mask[:, None],
    )


def flash_attn(q, k, theta, v, num_q_heads, num_kv_heads, scale=None):
    """q: [B, S, Hq*D], k: [B, S, Hkv*D] strided views, theta: [D/2],
    v: [B, Hkv, S, D] -> [B, S, Hq * D]. RoPE is applied inside the kernel."""
    b, s = q.shape[0], q.shape[1]
    d = q.shape[-1] // num_q_heads
    assert q.shape[-1] == num_q_heads * d and k.shape[-1] == num_kv_heads * d
    assert num_q_heads % num_kv_heads == 0
    if scale is None:
        scale = d ** -0.5

    out = _buf("flash", (b, s, num_q_heads * d), q.dtype, q.device)

    # Single-block softmax over the whole row (S is small in this workload):
    # avoids online rescaling so the rounding matches torch.softmax exactly.
    # BLOCK_N is padded to >= 16 to satisfy tl.dot's minimum extent, and
    # num_stages stays at 1 so the rope tiles fit in shared memory.
    block_m = 64
    block_n = max(16, triton.next_power_of_2(s))
    grid = (triton.cdiv(s, block_m), b * num_q_heads)
    _flash_attn_kernel[grid](
        q, k, v, theta, out,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1),
        H_Q=num_q_heads, H_KV=num_kv_heads, GROUP=num_q_heads // num_kv_heads,
        S=s, SCALE=scale, D=d, HALF=d // 2,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4 if block_n <= 32 else 8, num_stages=1,
    )
    return out


# ---------------------------------------------------------------------------
# Row argmax with first-max-wins tie semantics (matches torch.argmax).
# ---------------------------------------------------------------------------
@triton.jit
def _argmax_kernel(X, IDX, N, stride_x, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = X + row * stride_x

    best_val = tl.full((), float("-inf"), dtype=tl.float32)
    best_idx = tl.full((), 0, dtype=tl.int32)

    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(base + cols, mask=cols < N, other=float("-inf")).to(tl.float32)
        local_max = tl.max(x, axis=0)
        # Smallest index reaching the local max keeps first-occurrence order.
        local_idx = tl.min(tl.where(x == local_max, cols, N))
        better = local_max > best_val
        best_val = tl.where(better, local_max, best_val)
        best_idx = tl.where(better, local_idx, best_idx)

    tl.store(IDX + row, best_idx.to(tl.int64))


def argmax(logits):
    """logits: [..., N] -> int64 indices of shape [...], like torch.argmax."""
    x_2d = logits.reshape(-1, logits.shape[-1])
    rows, n = x_2d.shape
    idx = torch.empty(rows, dtype=torch.int64, device=logits.device)
    _argmax_kernel[(rows,)](x_2d, idx, n, x_2d.stride(0), BLOCK=8192, num_warps=8)
    return idx.view(logits.shape[:-1])


# ---------------------------------------------------------------------------
# Skinny GEMM (A [M,K] @ W^T [K,N]) tuned for the small-M regime of this
# workload (M = batch * seq grows ~64..192). cuBLAS underutilizes the GPU on
# these shapes; per-(N,K) tile/split-K configs were tuned on device.
# Shapes without a tuned config, or M outside the tuned range, fall back to
# cuBLAS via F.linear.
# ---------------------------------------------------------------------------
@triton.jit
def _skinny_gemm_kernel(
    A, B, C, WS,
    M, N: tl.constexpr, K,
    sam, sak, sbk, sbn, scm, scn, sws,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = N // BN
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = pid_k * BK + tl.arange(0, BK)

    a_ptrs = A + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = B + offs_k[:, None] * sbk + offs_n[None, :] * sbn

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, SPLIT_K * BK)):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += SPLIT_K * BK * sak
        b_ptrs += SPLIT_K * BK * sbk

    if SPLIT_K == 1:
        c_ptrs = C + offs_m[:, None] * scm + offs_n[None, :] * scn
        tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=offs_m[:, None] < M)
    else:
        ws_ptrs = WS + pid_k.to(tl.int64) * sws + offs_m[:, None] * N + offs_n[None, :]
        tl.store(ws_ptrs, acc, mask=offs_m[:, None] < M)


@triton.jit
def _gemm_reduce_kernel(WS, C, TOTAL, SCK, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL

    acc = tl.load(WS + offs, mask=mask, other=0.0)
    for k in range(1, SCK):
        acc += tl.load(WS + k.to(tl.int64) * TOTAL + offs, mask=mask, other=0.0)

    tl.store(C + offs, acc.to(C.dtype.element_ty), mask=mask)


# (N, K) -> (BM, BN, BK, SPLIT_K, num_warps, num_stages, min_M for the
# Triton path). Tuned on RTX 5090 against cuBLAS for M in ~64..192.
_GEMM_CONFIGS = {
    (16384, 2048): (64, 64, 64, 1, 8, 3, 0),     # fused gate+up projection
    (2048, 8192): (64, 128, 64, 4, 8, 3, 112),   # down projection
    (4096, 2048): (64, 64, 64, 4, 8, 3, 160),    # fused q/k/v projection
}
_TRITON_GEMM_MAX_M = 256


def linear(x, weight):
    """x @ weight.T, dispatching to the tuned Triton skinny GEMM for the
    small-M shapes of this workload and to cuBLAS (F.linear) otherwise."""
    shape = x.shape
    k = shape[-1]
    x_2d = x.reshape(-1, k)
    m = x_2d.shape[0]
    n = weight.shape[0]

    cfg = _GEMM_CONFIGS.get((n, k))
    if cfg is None or m < cfg[6] or m > _TRITON_GEMM_MAX_M:
        return torch.nn.functional.linear(x, weight)

    bm, bn, bk, split_k, num_warps, num_stages, _ = cfg
    out = _buf(f"gemm_{n}_{k}", (m, n), x.dtype, x.device)

    if split_k == 1:
        ws = out  # unused
        sws = 0
    else:
        ws = _buf(f"gemm_ws_{n}_{k}", (split_k * m * n,), torch.float32, x.device)
        sws = m * n

    grid = (triton.cdiv(m, bm) * (n // bn), split_k)
    _skinny_gemm_kernel[grid](
        x_2d, weight, out, ws,
        m, N=n, K=k,
        sam=x_2d.stride(0), sak=x_2d.stride(1),
        sbk=weight.stride(1), sbn=weight.stride(0),
        scm=out.stride(0), scn=out.stride(1), sws=sws,
        BM=bm, BN=bn, BK=bk, GROUP_M=8, SPLIT_K=split_k,
        num_warps=num_warps, num_stages=num_stages,
    )

    if split_k > 1:
        total = m * n
        block = 1024
        _gemm_reduce_kernel[(triton.cdiv(total, block),)](
            ws, out, total, SCK=split_k, BLOCK=block, num_warps=4,
        )

    return out.view(shape[:-1] + (n,))

