"""torch.compile'd single-step decode for the lean runtime (experiment).

Top-1 MoE routing is data-dependent, so the whole step cannot be one static
graph without either gathering expert weights (extra memory traffic) or
running all 7 experts. Instead each layer's dense part -- rmsnorm, QKV, RoPE,
KV-cache write, attention over the static cache with a position mask, O proj,
residual, second rmsnorm, router argmax -- is one compiled graph (shared by all
6 layers: weights are graph inputs, so no per-layer recompile), and the expert
FFN runs eagerly on the routed expert only.

Static shapes: batch B and cache length Tmax (the generator rounds Tmax up to
a bucket so a handful of graphs cover every request).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .lean import Fp32Linear, LeanModel, W8BfLinear

TMAX_BUCKET = 256


def _lin_fn(obj):
    """(kind, tensors) for a linear backend object, usable inside the graph."""
    if isinstance(obj, W8BfLinear):
        return "w8bf", (obj.q, obj.s)
    if isinstance(obj, Fp32Linear):
        return "fp32", (obj.w,)
    raise TypeError(f"compiled step does not support {type(obj).__name__}")


def _apply(kind, ws, x):
    if kind == "w8bf":
        shape = x.shape
        y = torch.ops.aten._weight_int8pack_mm(x.reshape(-1, shape[-1]).to(torch.bfloat16), ws[0], ws[1])
        return y.to(torch.float32).reshape(*shape[:-1], -1)
    return F.linear(x, ws[0])


def make_step(model: LeanModel, compile_mode: str | None = "default"):
    m = model
    H, D, eps, half = m.n_heads, m.head_dim, m.eps, m.head_dim // 2
    kinds = {_lin_fn(m.layers[0].qkv)[0]}
    assert len(kinds) == 1
    kind = kinds.pop()

    def dense(x, ln1, qkv_w, o_w, ln2, router, kc, vc, pos, cos, sin, mask):
        # x [B,1,hidden]; kc/vc [B,H,Tmax,D]; pos [1] long; mask [Tmax] bool
        B = x.shape[0]
        h = ln1 * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
        qkv = _apply(kind, qkv_w, h).view(B, 1, 3, H, D).permute(2, 0, 3, 1, 4)

        def rope(t):
            rot = torch.cat((-t[..., half:], t[..., :half]), dim=-1)
            return t * cos + rot * sin

        q, k = rope(qkv[0]), rope(qkv[1])
        kc.index_copy_(2, pos, k)
        vc.index_copy_(2, pos, qkv[2])
        a = F.scaled_dot_product_attention(q, kc, vc, attn_mask=mask, scale=m.scale)
        x = x + _apply(kind, o_w, a.transpose(1, 2).reshape(B, 1, H * D))
        h2 = ln2 * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))
        choice = F.linear(h2, router).argmax(-1)
        return x, h2, choice

    if compile_mode:
        import torch._dynamo.config as dc

        dc.recompile_limit = max(dc.recompile_limit, 64)
        kw = {} if compile_mode == "default" else {"mode": compile_mode}
        dense_c = torch.compile(dense, dynamic=False, fullgraph=True, **kw)
    else:
        dense_c = dense

    layer_args = [
        (L.ln1, _lin_fn(L.qkv)[1], _lin_fn(L.o)[1], L.ln2, L.router) for L in m.layers
    ]

    def step(tok: torch.Tensor, cur: int, cache) -> torch.Tensor:
        B = tok.shape[0]
        x = m.embed.index_select(0, tok).view(B, 1, -1)
        pos = torch.tensor([cur])
        cos = m.cos[cur : cur + 1]
        sin = m.sin[cur : cur + 1]
        mask = (torch.arange(cache.tmax) <= cur).view(1, 1, 1, -1)
        for li, L in enumerate(m.layers):
            ln1, qw, ow, ln2, router = layer_args[li]
            x, h2, choice = dense_c(x, ln1, qw, ow, ln2, router, cache.k[li], cache.v[li], pos, cos, sin, mask)
            x = x + m._moe_routed(L, h2.view(B, -1), choice.view(-1)).view(B, 1, -1)
        x = m._rms(x[:, -1], m.norm)
        return m.lm_head(x)

    step.tmax_bucket = TMAX_BUCKET
    return step
