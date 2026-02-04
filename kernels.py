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
    # 当前处理的行号
    row_idx = tl.program_id(0)
    
    # 计算当前行的起始指针
    row_start_ptr = X_ptr + row_idx * stride_x_row
    out_row_start_ptr = Out_ptr + row_idx * stride_y_row
    
    # 生成列的偏移量
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_COLS
    
    # 加载数据和权重，为了精度使用 float32 进行计算
    x_val = tl.load(row_start_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    w_val = tl.load(W_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # RMSNorm 计算逻辑
    # 1. 计算均方值
    x_sq = x_val * x_val
    mean_sq = tl.sum(x_sq, axis=0) / N_COLS
    
    # 2. 计算 rstd (reciprocal standard deviation)
    rstd = tl.rsqrt(mean_sq + eps)
    
    # 3. 归一化并应用 gamma 权重
    y_val = x_val * rstd * w_val
    
    # 写回显存
    tl.store(out_row_start_ptr + offsets, y_val, mask=mask)

def rms_norm_triton(x, weight, eps):
    # 将输入展平为 (Total_Rows, Hidden_Size)
    # 比如 (Batch, Seq_Len, Hidden) -> (Batch * Seq_Len, Hidden)
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    
    # 确保输入是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 展平以便获取正确的 stride
    x_flat = x.view(-1, N)
    
    # 分配输出空间
    y = torch.empty_like(x)
    
    # 计算 Block Size，向上取整到最近的 2 的幂次
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # Grid 维度：按照行数并行
    grid = (M,)
    
    _rms_norm_kernel[grid](
        x, weight, y,
        x_flat.stride(0),  # 输入的行 stride
        x_flat.stride(0),  # 输出的行 stride (通常和输入一样)
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
    实现旋转位置编码(Rotary Positional Encoding)的核心内核函数
    
    该函数对输入张量应用旋转位置编码变换，这是Transformer模型中一种重要的位置编码方法。
    通过将相邻维度配对并应用旋转变换，能够更好地捕捉序列中的相对位置信息。
    
    参数:
        X_ptr: 输入张量指针，形状为[batch, seq_len, num_heads, head_dim]
        Cos_ptr: 余弦值表指针，形状为[seq_len, head_dim//2]
        Sin_ptr: 正弦值表指针，形状为[seq_len, head_dim//2]
        stride_x_batch: 输入张量批次维度步长
        stride_x_seq: 输入张量序列维度步长
        stride_x_head: 输入张量注意力头维度步长
        stride_x_dim: 输入张量维度步长
        stride_c_seq: Cos/Sin表序列维度步长
        stride_c_dim: Cos/Sin表维度步长
        HEAD_DIM: 注意力头维度大小
        BLOCK_SIZE: 处理块大小，用于并行计算优化
        
    返回值:
        无返回值，直接在原地修改输入张量X_ptr
    """
    # 网格结构: (批次, 序列, 注意力头)
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    head_id = tl.program_id(2)

    # 计算当前注意力头的偏移量
    x_offset = (
        batch_id * stride_x_batch +
        seq_id * stride_x_seq +
        head_id * stride_x_head
    )
    
    # Cos/Sin仅依赖于序列位置(和维度)
    c_offset = seq_id * stride_c_seq
    
    # 旋转位置编码作用于注意力头维度的一半配对
    HALF_DIM = HEAD_DIM // 2
    
    # 我们并行处理范围[0, HALF_DIM)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HALF_DIM

    # 指向注意力头向量两个部分的指针
    # 通常最后一个维度是连续的所以stride_x_dim为1，但为了通用性我们使用stride
    # 索引方式: x[..., i] 和 x[..., i + HALF_DIM]
    x0_ptr = X_ptr + x_offset + offsets * stride_x_dim
    x1_ptr = X_ptr + x_offset + (offsets + HALF_DIM) * stride_x_dim
    
    c_ptr = Cos_ptr + c_offset + offsets * stride_c_dim
    s_ptr = Sin_ptr + c_offset + offsets * stride_c_dim # Sin表与Cos表布局相同
    
    # 加载数据
    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_ptr, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr, mask=mask, other=0.0).to(tl.float32)
    
    # 应用旋转变换
    # x0_new = x0 * cos - x1 * sin
    # x1_new = x0 * sin + x1 * cos
    y0 = x0 * c - x1 * s
    y1 = x0 * s + x1 * c
    
    # 存储回原位置(原地操作)
    tl.store(x0_ptr, y0, mask=mask)
    tl.store(x1_ptr, y1, mask=mask)

def apply_rotary_pos_emb_triton(x, cos, sin):
    # x: (Batch, Seq, Heads, Dim)
    # cos, sin: (Seq, Dim/2)
    # We perform the operation in-place on x.
    
    # Handle input continuity
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
    
    # Initialize output accumulator and statistics
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
    
    # Store output
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
    # They should be on CUDA and contiguous
    
    # Shape checks
    B, H, S, D = q.shape
    
    # Ensure inputs are contiguous
    if not q.is_contiguous(): q = q.contiguous()
    if not k.is_contiguous(): k = k.contiguous()
    if not v.is_contiguous(): v = v.contiguous()
       
    # Tuning block sizes
    BLOCK_M = 128
    BLOCK_N = 64
    
    # Grid
    # We parallelize over (Seq_Len // BLOCK_M, Batch, Heads)
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