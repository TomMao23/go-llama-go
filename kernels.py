import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_x_row, stride_y_row,
    N_COLS, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Current row index
    row_idx = tl.program_id(0)
    
    # Calculate start pointers for the current row
    row_start_ptr = X_ptr + row_idx * stride_x_row
    out_row_start_ptr = Out_ptr + row_idx * stride_y_row
    
    # Generate column offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_COLS
    
    # Load data and weights, use float32 for precision
    x_val = tl.load(row_start_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    w_val = tl.load(W_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # RMSNorm computation logic
    # 1. Calculate mean square
    x_sq = x_val * x_val
    mean_sq = tl.sum(x_sq, axis=0) / N_COLS
    
    # 2. Calculate rstd (reciprocal standard deviation)
    rstd = tl.rsqrt(mean_sq + eps)
    
    # 3. Normalize and apply gamma weight
    y_val = x_val * rstd * w_val
    
    # Write back to memory
    tl.store(out_row_start_ptr + offsets, y_val, mask=mask)

def rms_norm_triton(x, weight, eps):
    # Flatten input to (Total_Rows, Hidden_Size)
    # e.g., (Batch, Seq_Len, Hidden) -> (Batch * Seq_Len, Hidden)
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    
    # Ensure input is contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Flatten to get correct strides
    x_flat = x.view(-1, N)
    
    # Allocate output space
    y = torch.empty_like(x)
    
    # Calculate Block Size, round up to next power of 2
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # Grid dimensions: parallelize over rows
    grid = (M,)
    
    _rms_norm_kernel[grid](
        x, weight, y,
        x_flat.stride(0),  # Input row stride
        x_flat.stride(0),  # Output row stride (usually same as input)
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y
@triton.jit
def _rotary_kernel(
    X_ptr, Cos_ptr, Sin_ptr,
    stride_x_batch, stride_x_seq, stride_x_head, stride_x_dim,
    stride_c_seq, stride_c_dim,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Core kernel function for Rotary Positional Encoding (RoPE).

    This function applies rotary positional encoding transformation to the input tensor.
    By pairing adjacent dimensions and applying rotation, it captures relative position information in the sequence.

    Args:
        X_ptr: Pointer to input tensor, shape [batch, seq_len, num_heads, head_dim]
        Cos_ptr: Pointer to cosine table, shape [seq_len, head_dim//2]
        Sin_ptr: Pointer to sine table, shape [seq_len, head_dim//2]
        stride_x_batch: Stride for batch dimension of input tensor
        stride_x_seq: Stride for sequence dimension of input tensor
        stride_x_head: Stride for head dimension of input tensor
        stride_x_dim: Stride for dimension of input tensor
        stride_c_seq: Stride for sequence dimension of Cos/Sin table
        stride_c_dim: Stride for dimension of Cos/Sin table
        HEAD_DIM: Size of the head dimension
        BLOCK_SIZE: Block size for parallel computation optimization

    Returns:
        None, modified in-place
    """
    # Grid structure: (Batch, Seq, Heads)
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    head_id = tl.program_id(2)

    # Calculate offsets for the current head
    x_offset = (
        batch_id * stride_x_batch +
        seq_id * stride_x_seq +
        head_id * stride_x_head
    )
    
    # Cos/Sin depend only on sequence position (and dim)
    c_offset = seq_id * stride_c_seq
    
    # RoPE applies to pairs in the head dimension
    HALF_DIM = HEAD_DIM // 2
    
    # Process range [0, HALF_DIM) in parallel
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HALF_DIM

    # Pointers to the two halves of the head vector
    # Usually last dim is contiguous (stride=1), but use stride for generality
    # Indexing: x[..., i] and x[..., i + HALF_DIM]
    x0_ptr = X_ptr + x_offset + offsets * stride_x_dim
    x1_ptr = X_ptr + x_offset + (offsets + HALF_DIM) * stride_x_dim
    
    c_ptr = Cos_ptr + c_offset + offsets * stride_c_dim
    s_ptr = Sin_ptr + c_offset + offsets * stride_c_dim # Sin table has same layout as Cos
    
    # Load data
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_ptr, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr, mask=mask, other=0.0).to(tl.float32)
    
    # Apply rotation transformation
    # x0_new = x0 * cos - x1 * sin
    # x1_new = x0 * sin + x1 * cos
    y0 = x0 * c - x1 * s
    y1 = x0 * s + x1 * c
    
    # Store back to original location (in-place)
    tl.store(x0_ptr, y0, mask=mask)
    tl.store(x1_ptr, y1, mask=mask)

def apply_rotary_pos_emb_triton(x, cos, sin):
    # x: (Batch, Seq, Heads, Dim)
    # cos, sin: (Seq, Dim/2)
    # We perform the operation in-place on x
    
    # Ensure input is contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()
        
    B, S, H, D = x.shape
    
    # Block size >= HALF_DIM
    HALF_DIM = D // 2
    BLOCK_SIZE = triton.next_power_of_2(HALF_DIM)
    
    # Grid: (Batch, Seq, Heads)
    grid = (B, S, H)
    
    _rotary_kernel[grid](
        x, cos, sin,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        HEAD_DIM=D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return x

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale,
    L, # Sequence length
    Out,
    stride_q_batch, stride_q_head, stride_q_m, stride_q_k,
    stride_k_batch, stride_k_head, stride_k_n, stride_k_k,
    stride_v_batch, stride_v_head, stride_v_n, stride_v_k,
    stride_o_batch, stride_o_head, stride_o_m, stride_o_n,
    Z, H, N_CTX, # Batch, Heads, Context Length
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    # Grid: (Tr, B, H)
    start_m = tl.program_id(0)
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)

    # Offsets for Q, K, V pointers
    # shape: (B, H, S, D)
    q_offset = off_b * stride_q_batch + off_h * stride_q_head
    k_offset = off_b * stride_k_batch + off_h * stride_k_head
    v_offset = off_b * stride_v_batch + off_h * stride_v_head
    o_offset = off_b * stride_o_batch + off_h * stride_o_head

    # Block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_q_m, stride_q_k),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, N_CTX), # Transposed for Q @ K.T
        strides=(stride_k_k, stride_k_n),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_v_n, stride_v_k),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    # Initialize
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Load Q
    # We assume N_CTX is multiple of BLOCK_M/N or handle padding if needed. 
    # For this assignment, simple boundary check might be enough but block_ptr helps.
    q = tl.load(Q_block_ptr, boundary_check=(0, 1))
    
    # Loop over K, V blocks
    # Causal Masking: we only attend to keys up to the current query position
    # The end of the K loop is determined by (start_m + 1) * BLOCK_M
    
    # We loop from 0 to (start_m + 1) * BLOCK_M
    lo = 0
    hi = (start_m + 1) * BLOCK_M
    
    for start_n in range(lo, hi, BLOCK_N):
        # Load K, V
        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(0, 1))
        
        # Compute Q @ K.T
        qk = tl.dot(q, k)
        
        # Apply scaling
        qk *= sm_scale
        
        # Apply Causal Mask
        # If the block is on the diagonal (start_n == start_m * BLOCK_M), we mask
        if start_n + BLOCK_N > start_m * BLOCK_M:
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(mask, qk, float("-inf"))
            
        # Online Softmax update
        m_i_new = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_i_new)
        
        p = tl.exp(qk - m_i_new[:, None])
        alpha = tl.exp(m_i - m_i_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        
        # acc update
        # acc = acc * alpha + p @ v
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v) # Precision handling
        
        # Update statistics
        m_i = m_i_new
        
        # Advance pointers 
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # Final normalization
    # Check for division by zero (e.g. fully masked blocks?)
    # l_i should be > 0 if any attention was valid.
    acc = acc / l_i[:, None]
    
    # Store Output
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_o_m, stride_o_n),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))

def flash_attention_triton(q, k, v):
    # q, k, v: (Batch, Heads, Seq_Len, Dim)
    
    # Shape checks
    B, H, S, D = q.shape
    
    # Ensure contiguous
    if not q.is_contiguous(): q = q.contiguous()
    if not k.is_contiguous(): k = k.contiguous()
    if not v.is_contiguous(): v = v.contiguous()
       
    # block sizes
    BLOCK_M = 128
    BLOCK_N = 64
    
    # Grid
    # Parallelize over (Seq_Len // BLOCK_M, Batch, Heads)
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    scale = 1.0 / (D ** 0.5)
    
    # Initialize output
    o = torch.empty_like(q)
    
    _flash_attn_fwd_kernel[grid](
        q, k, v, scale,
        S,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, S,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D
    )
    
    return o