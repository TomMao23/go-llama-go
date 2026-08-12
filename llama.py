import dataclasses
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

import triton_kernels as tk


@dataclasses.dataclass
class ModelConfig:
    head_dim: int

    hidden_size: int

    intermediate_size: int

    num_attention_heads: int

    num_hidden_layers: int

    num_key_value_heads: int

    rms_norm_eps: float

    rope_theta: float

    torch_dtype: str

    vocab_size: int


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))

        self.eps = eps

    def forward(self, input):
        return tk.rms_norm(input, self.weight, self.eps)


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.silu = nn.SiLU()

    def forward(self, input):
        # gate_up_proj is the fused gate+up GEMM built by
        # ModelForCausalLM.fuse_projections.
        return self.down_proj(tk.swiglu(self.gate_up_proj(input)))


def apply_rotary_position_embedding(input, sin_table, cos_table):
    # Kept only as a numeric reference; the hot path uses the fused Triton
    # kernel `tk.rope`, which also folds in the [B,S,H,D] -> [B,H,S,D]
    # permutation.
    sin_table = sin_table[None, :, None, :]
    cos_table = cos_table[None, :, None, :]

    input_0 = input[..., : input.shape[-1] // 2]
    input_1 = input[..., input.shape[-1] // 2 :]
    input_0_rotated = input_0 * cos_table - input_1 * sin_table
    input_1_rotated = input_0 * sin_table + input_1 * cos_table

    return torch.cat((input_0_rotated, input_1_rotated), dim=-1)


def apply_scaled_dot_product_attention(query, key, value):
    # Kept only as a numeric reference; the hot path uses the fused Triton
    # flash attention kernel `tk.flash_attn` (native GQA, causal mask inline,
    # no S x S score matrix materialized).
    _, num_heads_q, seq_len_q, emb_dim = query.shape
    _, num_heads_k, seq_len_k, _ = key.shape
    _, num_heads_v, _, _ = value.shape

    key = key.repeat_interleave(num_heads_q // num_heads_k, 1)
    value = value.repeat_interleave(num_heads_q // num_heads_v, 1)

    scale = 1 / math.sqrt(emb_dim)
    attn_mask = torch.tril(
        torch.full((seq_len_q, seq_len_k), True, device=query.device)
    )

    attn_output = torch.matmul(query, key.permute(0, 1, 3, 2)) * scale
    attn_output = torch.where(attn_mask, attn_output, float("-inf"))
    attn_output = torch.softmax(attn_output, dim=-1)
    attn_output = torch.matmul(attn_output, value)

    return attn_output


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.head_dim

        self.hidden_size = config.hidden_size

        self.num_attention_heads = config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads

        self.rope_theta = config.rope_theta

        # Kept for checkpoint loading; ModelForCausalLM.fuse_projections
        # replaces them with a single fused qkv_proj GEMM.
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )

        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(self, hidden_states):
        batch_size, seq_len = hidden_states.shape[:2]

        # One fused q/k/v GEMM; the q/k/v slices are consumed as strided
        # views, so no intermediate copy is needed.
        qkv = self.qkv_proj(hidden_states)
        q_dim = self.num_attention_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        query_slice = qkv[..., :q_dim]
        key_slice = qkv[..., q_dim : q_dim + kv_dim]
        value_slice = qkv[..., q_dim + kv_dim :]

        # Strided [B, Hkv, S, D] view over the v slice for the attention kernel.
        value_states = torch.as_strided(
            value_slice,
            (batch_size, self.num_key_value_heads, seq_len, self.head_dim),
            (value_slice.stride(0), self.head_dim, value_slice.stride(1), 1),
        )

        if seq_len > 256:
            # Safety net for very long sequences beyond the flash kernel's
            # single-block softmax design: fall back to the reference path.
            sin_table, cos_table = generate_sin_and_cos_tables(
                seq_len, self.head_dim, base=self.rope_theta,
                dtype=query_slice.dtype, device=hidden_states.device,
            )
            query_states = apply_rotary_position_embedding(
                query_slice.reshape(batch_size, seq_len, -1, self.head_dim),
                sin_table, cos_table,
            ).permute(0, 2, 1, 3)
            key_states = apply_rotary_position_embedding(
                key_slice.reshape(batch_size, seq_len, -1, self.head_dim),
                sin_table, cos_table,
            ).permute(0, 2, 1, 3)
            attn_output = apply_scaled_dot_product_attention(
                query_states, key_states, value_states
            )
            return self.o_proj(
                attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
            )

        # Fused RoPE + causal flash attention with native GQA in one kernel;
        # the output lands in [B, S, H * D] layout, ready for o_proj.
        attn_output = tk.flash_attn(
            query_slice, key_slice, self.theta, value_states,
            self.num_attention_heads, self.num_key_value_heads,
        )

        return self.o_proj(attn_output)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.self_attn = Attention(config)

        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def fused_forward(self, residual, normed_hidden_states):
        # Fused residual stream: takes the un-normalized residual and the
        # already-normalized input, and returns the updated pair, with every
        # "residual add + RMSNorm" collapsed into one Triton kernel.
        attn_output = self.self_attn(normed_hidden_states)
        residual, normed_hidden_states = tk.add_rms_norm(
            residual, attn_output, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps
        )

        mlp_output = self.mlp(normed_hidden_states)
        residual, normed_hidden_states = tk.add_rms_norm(
            residual, mlp_output, self.next_norm_weight, self.input_layernorm.eps
        )

        return residual, normed_hidden_states


def generate_sin_and_cos_tables(seq_len, emb_dim, base, dtype, device):
    theta = base ** (
        -2 * (torch.arange(emb_dim // 2, dtype=dtype, device=device) / emb_dim)
    )

    positions = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(1)
    sin_table = torch.sin(positions * theta)
    cos_table = torch.cos(positions * theta)

    return sin_table, cos_table


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.head_dim

        self.hidden_size = config.hidden_size

        self.num_hidden_layers = config.num_hidden_layers

        self.rms_norm_eps = config.rms_norm_eps

        self.rope_theta = config.rope_theta

        self.torch_dtype = config.torch_dtype

        self.vocab_size = config.vocab_size

        self.embed_tokens = torch.nn.Embedding(self.vocab_size, self.hidden_size)

        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(self.num_hidden_layers)
        )

        self.norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)

        # Drive the residual stream with fused add+RMSNorm kernels: after the
        # loop, normed_hidden_states already equals self.norm(hidden_states),
        # so the final RMSNorm comes for free.
        residual = hidden_states
        normed_hidden_states = self.layers[0].input_layernorm(hidden_states)

        for i in range(self.num_hidden_layers):
            residual, normed_hidden_states = self.layers[i].fused_forward(
                residual, normed_hidden_states
            )

        return normed_hidden_states


class ModelForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = Model(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=20):
        for _ in range(max_new_tokens):
            hidden_states = self.model(input_ids)

            logits = self.lm_head(hidden_states[:, -1, :])

            next = tk.argmax(logits).unsqueeze(-1)

            input_ids = torch.cat((input_ids, next), dim=-1)

        return input_ids

    @staticmethod
    def from_pretrained(model_path):
        model_path = Path(model_path)

        with open(model_path / "config.json") as f:
            config = json.load(f)

        if "head_dim" not in config:
            config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]

        config = ModelConfig(
            **{
                key: value
                for key, value in config.items()
                if key in ModelConfig.__annotations__
            }
        )

        model = ModelForCausalLM(config).to(getattr(torch, config.torch_dtype))

        state_dict = load_file(model_path / "model.safetensors")

        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        model.load_state_dict(state_dict)

        model.fuse_projections()

        # RoPE per-dim frequencies, computed once with the exact same bf16
        # arithmetic as the baseline's sin/cos table generation.
        dtype = getattr(torch, config.torch_dtype)
        theta = config.rope_theta ** (
            -2
            * (
                torch.arange(config.head_dim // 2, dtype=dtype)
                / config.head_dim
            )
        )
        for layer in model.model.layers:
            layer.self_attn.register_buffer("theta", theta)

        return model

    def fuse_projections(self):
        # Weight-level fusion feeding the Triton kernels: q/k/v into one GEMM
        # and gate/up into one GEMM. Also wires the residual-stream norm
        # weights used by DecoderLayer.fused_forward.
        layers = self.model.layers

        for i, layer in enumerate(layers):
            attn = layer.self_attn
            qkv_weight = torch.cat(
                (attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight), dim=0
            )
            attn.qkv_proj = nn.Linear(
                attn.hidden_size, qkv_weight.shape[0], bias=False,
                device=attn.q_proj.weight.device, dtype=attn.q_proj.weight.dtype,
            )
            attn.qkv_proj.weight = nn.Parameter(qkv_weight)
            # Release the separate projections; qkv_proj replaces them.
            del attn.q_proj, attn.k_proj, attn.v_proj

            mlp = layer.mlp
            gate_up_weight = torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight), dim=0)
            mlp.gate_up_proj = nn.Linear(
                mlp.gate_proj.in_features, gate_up_weight.shape[0], bias=False,
                device=mlp.gate_proj.weight.device, dtype=mlp.gate_proj.weight.dtype,
            )
            mlp.gate_up_proj.weight = nn.Parameter(gate_up_weight)
            del mlp.gate_proj, mlp.up_proj

            next_norm_weight = (
                layers[i + 1].input_layernorm.weight
                if i + 1 < len(layers)
                else self.model.norm.weight
            )
            layer.next_norm_weight = next_norm_weight
