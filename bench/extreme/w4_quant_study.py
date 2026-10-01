"""Lower-bit weight study for the live booper model (track w4, phase 1).

Torch-only fake quantization of the int8 snapshot: every variant starts from
the on-disk int8 weights (q * scale, the thing that serves today) and
re-quantizes some matrices to int4/5/6, group-wise along the input dim, with
plain round-to-nearest (RTN) or GPTQ error-compensated rounding.

    python bench/extreme/w4_quant_study.py data      # eval + calibration sets (public HF data)
    python bench/extreme/w4_quant_study.py ref       # fixture ref + int8 eval stats + hidden states
    python bench/extreme/w4_quant_study.py variant NAME [NAME...]
    python bench/extreme/w4_quant_study.py head      # two-stage lm_head screen study

Everything is written under $W4_DIR (default /tmp/maxperf/w4). The eval data
are public datasets (Discord-Dialogues train rows, ultrachat_200k test_sft),
fetched as parquet and pre-extracted to jsonl by extract_data.py; nothing here
reads the consented corpus.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sft"))

MODEL_DIR = Path(os.environ.get("W4_MODEL_DIR", "/home/jason/projects/babble/artifacts/hf-booper-longctx-v1"))
W = Path(os.environ.get("W4_DIR", "/tmp/maxperf/w4"))
THREADS = int(os.environ.get("W4_THREADS", "3"))
torch.set_num_threads(THREADS)

N_DISCORD_EVAL, N_ULTRA_EVAL = 200, 100
N_DISCORD_CAL, N_ULTRA_CAL = 48, 16
SEQ_CAP = 1024  # tokens per eval sequence (prompt fitted to the 1536 budget, then capped)


# ----------------------------------------------------------------- data ---
def _tok():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))


def build_data() -> None:
    from babble.conversation import conversation_prompt_for_token_budget
    from sft_longform import _chatml_turns, _messages_group, _turn_records

    tok = _tok()
    bos, sep, eos = (tok.token_to_id(t) for t in ("<bos>", "<sep>", "<eos>"))
    count = lambda s: len(tok.encode(s, add_special_tokens=False).ids)  # noqa: E731

    def encode(record):
        r = tok.encode(record.response, add_special_tokens=False).ids[:256]
        if len(r) < 2:
            return None
        budget = min(1536, SEQ_CAP - 3 - len(r))
        if budget < 16:
            return None
        prompt = conversation_prompt_for_token_budget(
            record.history, record.current_user, max_turns=8, max_chars=0, max_tokens=budget, token_count=count
        )
        p = tok.encode(prompt, add_special_tokens=False).ids
        ids = [bos, *p, sep, *r, eos]
        return {"ids": ids, "sep_at": 1 + len(p)}

    def records(path, kind):
        for line in open(path):
            row = json.loads(line)
            if kind == "discord":
                recs = _turn_records("discord", "x", _chatml_turns(row["text"]))
            else:
                recs = _messages_group("ultrachat", row["messages"])
            if recs:
                yield recs[-1]  # the target with the most history

    out = {"eval": [], "cal": []}
    for path, kind, n_eval, n_cal in (
        (W / "discord.jsonl", "discord", N_DISCORD_EVAL, N_DISCORD_CAL),
        (W / "ultrachat.jsonl", "ultrachat", N_ULTRA_EVAL, N_ULTRA_CAL),
    ):
        got = []
        for rec in records(path, kind):
            e = encode(rec)
            if e:
                e["src"] = kind
                got.append(e)
            if len(got) >= n_eval + n_cal:
                break
        out["eval"] += got[:n_eval]
        out["cal"] += got[n_eval : n_eval + n_cal]
    for k, v in out.items():
        print(k, len(v), "seqs", sum(len(e["ids"]) for e in v), "tokens",
              "resp tokens", sum(len(e["ids"]) - e["sep_at"] - 1 for e in v))
    torch.save(out, W / "evalset.pt")


# ---------------------------------------------------------------- model ---
def load_packed():
    from safetensors.torch import load_file

    return load_file(str(MODEL_DIR / "model-int8.safetensors"))


def build_model(packed, override: dict[str, torch.Tensor] | None = None):
    """MixtralForCausalLM fp32 from the int8 pack; `override` maps packed names -> fp32 weights."""
    from transformers import MixtralConfig, MixtralForCausalLM

    override = override or {}

    def unpack(name):
        if name in override:
            return override[name].float()
        v = packed[name]
        if v.dtype == torch.int8:
            return v.float() * packed[name + ".scale"].float()
        return v.float()

    config = MixtralConfig.from_pretrained(MODEL_DIR)
    model = MixtralForCausalLM(config)
    state = {}
    for name in packed:
        if name.endswith(".scale") or ".block_sparse_moe." in name or name == "lm_head.weight":
            continue
        state[name] = unpack(name)
    fused = True
    for layer in range(config.num_hidden_layers):
        old = f"model.layers.{layer}.block_sparse_moe"
        new = f"model.layers.{layer}.mlp"
        state[new + ".gate.weight"] = unpack(old + ".gate.weight")
        gu, dn = [], []
        for e in range(config.num_local_experts):
            p = f"{old}.experts.{e}"
            gu.append(torch.cat((unpack(p + ".w1.weight"), unpack(p + ".w3.weight"))))
            dn.append(unpack(p + ".w2.weight"))
        state[new + ".experts.gate_up_proj"] = torch.stack(gu)
        state[new + ".experts.down_proj"] = torch.stack(dn)
    assert fused
    model.load_state_dict(state, strict=False, assign=True)
    # tied head: lm_head shares embed_tokens (the engine reuses that panel too)
    head = override.get("lm_head.weight")
    if head is None:
        model.lm_head.weight = model.model.embed_tokens.weight
    else:
        # untie: embedding keeps its int8 weights, the head gets the variant
        model.lm_head.weight = torch.nn.Parameter(head.float())
    return model.eval(), config


# -------------------------------------------------------------- quantizers ---
LINEAR_NAMES = None


def weight_names(packed, which: str) -> list[str]:
    """Packed int8 matrix names in a group: attn | experts | head | router."""
    out = []
    for n, v in packed.items():
        if v.dtype != torch.int8:
            continue
        if which == "attn" and ".self_attn." in n:
            out.append(n)
        elif which == "experts" and ".experts." in n:
            out.append(n)
        elif which == "head" and n == "model.embed_tokens.weight":
            out.append(n)
    return sorted(out)


def int8_fp32(packed, name):
    return packed[name].float() * packed[name + ".scale"].float()


def _qparams(w: torch.Tensor, bits: int, sym: bool, clip: float = 1.0):
    """Scale/zero for rows of `w` [..., g] (last dim = group)."""
    if sym:
        qmax = 2 ** (bits - 1) - 1
        amax = w.abs().amax(-1, keepdim=True).clamp_min(1e-12) * clip
        s = (amax / qmax).half().float().clamp_min(1e-7)
        return s, None, -qmax - 1, qmax
    lo = w.amin(-1, keepdim=True) * clip
    hi = w.amax(-1, keepdim=True) * clip
    qmax = 2**bits - 1
    s = ((hi - lo) / qmax).half().float().clamp_min(1e-7)
    z = torch.round(-lo / s).clamp(0, qmax)
    return s, z, 0, qmax


def _fq(w, s, z, lo, hi):
    if z is None:
        return torch.clamp(torch.round(w / s), lo, hi) * s
    return (torch.clamp(torch.round(w / s) + z, lo, hi) - z) * s


def rtn(w: torch.Tensor, bits: int, g: int, sym: bool, mse_clip: bool = False) -> torch.Tensor:
    out_f, in_f = w.shape
    wg = w.reshape(out_f, in_f // g, g)
    if not mse_clip:
        s, z, lo, hi = _qparams(wg, bits, sym)
        return _fq(wg, s, z, lo, hi).reshape(out_f, in_f)
    best = None
    best_err = None
    for clip in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75):
        s, z, lo, hi = _qparams(wg, bits, sym, clip)
        q = _fq(wg, s, z, lo, hi)
        err = (q - wg).pow(2).sum(-1, keepdim=True)
        if best is None:
            best, best_err = q, err
        else:
            m = err < best_err
            best = torch.where(m, q, best)
            best_err = torch.where(m, err, best_err)
    return best.reshape(out_f, in_f)


def gptq(w: torch.Tensor, H: torch.Tensor, bits: int, g: int, sym: bool, damp: float = 0.01,
         act_order: bool = False, blocksize: int = 128) -> torch.Tensor:
    """GPTQ (Frantar et al.) with group-wise scales computed on the error-updated weights."""
    W_ = w.clone().double()
    H = H.clone().double()
    n = H.shape[0]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W_[:, dead] = 0
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W_ = W_[:, perm]
        H = H[perm][:, perm]
    H += damp * torch.mean(torch.diag(H)) * torch.eye(n, dtype=H.dtype)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W_)
    # with act_order, groups follow the *original* column order: precompute
    # params on the fly per original group the first time a column of it is hit
    gparams: dict[int, tuple] = {}
    for i1 in range(0, n, blocksize):
        i2 = min(i1 + blocksize, n)
        W1 = W_[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            orig = int(perm[col]) if perm is not None else col
            gi = orig // g
            if gi not in gparams:
                if perm is None:
                    # current (error-updated) group columns; those left of i1+i inside the block
                    # are already quantized, but group starts align with blocks (g | blocksize)
                    grp = torch.cat((W1[:, i:], W_[:, i2:]), 1)[:, :g]
                else:
                    cols = (perm >= gi * g) & (perm < (gi + 1) * g)
                    idx = torch.nonzero(cols).flatten()
                    full = torch.cat((W_[:, :i1], W1, W_[:, i2:]), 1)
                    grp = full[:, idx]
                gparams[gi] = _qparams(grp.float(), bits, sym)
            s, z, lo, hi = gparams[gi]
            wcol = W1[:, i]
            q = _fq(wcol.float().unsqueeze(1), s, z, lo, hi).squeeze(1).double()
            Q1[:, i] = q
            err = (wcol - q) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W_[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    if perm is not None:
        inv = torch.argsort(perm)
        Q = Q[:, inv]
    return Q.float()


# --------------------------------------------------------------- hessians ---
def collect_hessians(packed, seqs, model=None):
    """X^T X per input site from the int8 model: attn-in, o-in, expert-in, expert-mid, head-in."""
    model, config = (model, None) if model is not None else build_model(packed)
    L = len(model.model.layers)
    d = model.config.hidden_size
    E = model.config.num_local_experts
    ffn = model.config.intermediate_size
    H = {}
    cnt = {}

    def add(key, x):
        x = x.reshape(-1, x.shape[-1]).double()
        if key not in H:
            H[key] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64)
            cnt[key] = 0
        H[key] += x.T @ x
        cnt[key] += x.shape[0]

    hooks = []
    for l, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.q_proj.register_forward_pre_hook(lambda m, a, l=l: add(("attn_in", l), a[0])))
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: add(("o_in", l), a[0])))

        def moe_hook(m, a, l=l, layer=layer):
            x = a[0].reshape(-1, d)
            logits = layer.mlp.gate(x) if hasattr(layer.mlp.gate, "weight") and isinstance(layer.mlp.gate, torch.nn.Linear) else x @ layer.mlp.gate.weight.T
            sel = logits.argmax(-1)
            gu = layer.mlp.experts.gate_up_proj
            for e in range(E):
                xe = x[sel == e]
                if xe.shape[0] == 0:
                    continue
                add(("exp_in", l, e), xe)
                h = xe @ gu[e].T
                g_, u_ = h[:, :ffn], h[:, ffn:]
                add(("exp_mid", l, e), F.silu(g_) * u_)

        hooks.append(layer.mlp.register_forward_pre_hook(moe_hook))
    hooks.append(model.lm_head.register_forward_pre_hook(lambda m, a: add(("head_in",), a[0])))
    with torch.inference_mode():
        for s in seqs:
            model(torch.tensor([s["ids"]]))
    for h in hooks:
        h.remove()
    return {k: (v / cnt[k]).float() for k, v in H.items()}, cnt


def hkey_for(name: str):
    parts = name.split(".")
    if name == "model.embed_tokens.weight":
        return ("head_in",)
    l = int(parts[2])
    if ".self_attn." in name:
        return ("o_in", l) if parts[4] == "o_proj" else ("attn_in", l)
    e = int(parts[5])
    return ("exp_mid", l, e) if parts[6] == "w2" else ("exp_in", l, e)


# ---------------------------------------------------------------- variants ---
def parse_variant(name: str) -> dict:
    """e.g. 'all-q4g64s-rtn', 'exp-q4g128a-gptq', 'noh-q5g64s-rtn', 'head-q4g32s-rtn'.

    scope: all (attn+experts+head) | noh (attn+experts) | exp (experts) | attn | head
    q{bits}g{group}{s|a}  sym/asym; method rtn | rtnc (mse clip) | gptq | gptqa (act order)
    """
    scope, q, method = name.split("-")
    bits = int(q[1 : q.index("g")])
    g = int(q[q.index("g") + 1 : -1])
    sym = q[-1] == "s"
    return {"scope": scope, "bits": bits, "g": g, "sym": sym, "method": method}


SCOPES = {
    "all": ("attn", "experts", "head"),
    "noh": ("attn", "experts"),
    "exp": ("experts",),
    "attn": ("attn",),
    "head": ("head",),
}


def make_override(packed, spec, hess=None) -> dict[str, torch.Tensor]:
    over = {}
    for grp in SCOPES[spec["scope"]]:
        for n in weight_names(packed, grp):
            w = int8_fp32(packed, n)
            if spec["method"] in ("rtn", "rtnc"):
                q = rtn(w, spec["bits"], spec["g"], spec["sym"], mse_clip=spec["method"] == "rtnc")
            else:
                Hm = hess.get(hkey_for(n))
                if Hm is None:  # expert never routed in calibration
                    q = rtn(w, spec["bits"], spec["g"], spec["sym"])
                else:
                    q = gptq(w, Hm, spec["bits"], spec["g"], spec["sym"], act_order=spec["method"] == "gptqa")
            # tied embedding: quantizing the head quantizes the shared table
            over["lm_head.weight" if n == "model.embed_tokens.weight" else n] = q
            if n == "model.embed_tokens.weight":
                over["lm_head.weight"] = q
    return over


# ----------------------------------------------------------------- eval ---
def fixture_ref_path():
    return W / "ref-longctx.pt"


def build_fixture_ref(model):
    import bench.extreme.reference as ref

    tok = _tok()
    out = []
    with torch.inference_mode():
        for prompt, response in ref.CASES:
            ids, sep_at = ref.encode_case(tok, prompt, response)
            logits = model(torch.tensor([ids])).logits[0].float()
            out.append({"ids": ids, "sep_at": sep_at, "logits": logits.half()})
    torch.save({"model_dir": str(MODEL_DIR), "cases": out}, fixture_ref_path())


def fixture_compare(model) -> dict:
    import bench.extreme.reference as ref

    ref.REF_PATH = fixture_ref_path()

    def fn(ids):
        with torch.inference_mode():
            return model(torch.tensor([ids])).logits[0].float()

    return ref.compare(fn)


def eval_stats(model, seqs, keep_hidden: int = 0, seed: int = 0):
    """Per-sequence target logprobs (response), argmax and top-40 ids (all positions)."""
    rng = random.Random(seed)
    stats = []
    hidden = []
    captured = {}
    hook = model.lm_head.register_forward_pre_hook(lambda m, a: captured.__setitem__("h", a[0][0].detach()))
    with torch.inference_mode():
        for s in seqs:
            ids = s["ids"]
            logits = model(torch.tensor([ids])).logits[0].float()
            lp = logits.log_softmax(-1)
            tgt = torch.tensor(ids[1:])
            tlp = lp[:-1].gather(1, tgt.unsqueeze(1)).squeeze(1)
            top = logits.topk(40, -1).indices.short()
            stats.append({"tlp": tlp, "argmax": logits.argmax(-1), "top40": top})
            if keep_hidden:
                pos = list(range(len(ids)))
                pick = rng.sample(pos, min(keep_hidden, len(pos)))
                for p in pick:
                    hidden.append((captured["h"][p].clone(), ids[: p + 1]))
    hook.remove()
    return stats, hidden


def eval_compare(ref_stats, cand_stats, seqs) -> dict:
    agree = total = 0
    dn = {"all": [0.0, 0.0, 0], "discord": [0.0, 0.0, 0], "ultrachat": [0.0, 0.0, 0]}
    top40_overlap = 0.0
    kl_like = 0.0
    for r, c, s in zip(ref_stats, cand_stats, seqs):
        agree += int((r["argmax"] == c["argmax"]).sum())
        total += r["argmax"].numel()
        a = s["sep_at"]
        rn = -float(r["tlp"][a:].sum())
        cn = -float(c["tlp"][a:].sum())
        n = len(s["ids"]) - 1 - a
        for k in ("all", s["src"]):
            dn[k][0] += rn
            dn[k][1] += cn
            dn[k][2] += n
        ov = (r["top40"].unsqueeze(-1) == c["top40"].unsqueeze(-2)).any(-1).float().mean(-1)
        top40_overlap += float(ov.sum())
        kl_like += float((r["tlp"] - c["tlp"]).sum())
    out = {"top1_agreement": agree / total, "positions": total, "top40_overlap": top40_overlap / total}
    for k, (rn, cn, n) in dn.items():
        out[f"ref_nll_{k}"] = rn / n
        out[f"dnll_{k}"] = (cn - rn) / n
        out[f"resp_tokens_{k}"] = n
    out["dnll_alltok"] = kl_like / sum(len(s["ids"]) - 1 for s in seqs)
    return out


def cmd_ref():
    packed = load_packed()
    model, _ = build_model(packed)
    build_fixture_ref(model)
    print("fixture self-check", fixture_compare(model))
    data = torch.load(W / "evalset.pt")
    t = time.time()
    stats, hidden = eval_stats(model, data["eval"], keep_hidden=24)
    print(f"eval {time.time() - t:.1f}s, hidden {len(hidden)}")
    torch.save(stats, W / "ref-eval-stats.pt")
    torch.save(hidden, W / "ref-hidden.pt")
    t = time.time()
    hess, cnt = collect_hessians(packed, data["cal"] + [
        {"ids": __import__("bench.extreme.reference", fromlist=["x"]).encode_case(_tok(), p, r)[0]}
        for p, r in __import__("bench.extreme.reference", fromlist=["x"]).CASES
    ], model)
    print(f"hessians {time.time() - t:.1f}s; tokens per site min",
          min(cnt.values()), "max", max(cnt.values()), "sites", len(cnt))
    torch.save({"H": hess, "cnt": cnt}, W / "hessians.pt")


def cmd_variant(names):
    packed = load_packed()
    data = torch.load(W / "evalset.pt")
    ref_stats = torch.load(W / "ref-eval-stats.pt")
    hess = None
    results_path = W / "variants.jsonl"
    for name in names:
        spec = parse_variant(name)
        if spec["method"].startswith("gptq") and hess is None:
            hd = torch.load(W / "hessians.pt")
            # rarely-routed experts: too few calibration tokens for a Hessian -> RTN
            hess = {k: v for k, v in hd["H"].items() if hd["cnt"][k] >= 128}
        t = time.time()
        over = make_override(packed, spec, hess)
        tq = time.time() - t
        model, _ = build_model(packed, over)
        fx = fixture_compare(model)
        cand_stats, _ = eval_stats(model, data["eval"])
        ev = eval_compare(ref_stats, cand_stats, data["eval"])
        row = {"variant": name, **spec, "quant_s": round(tq, 1), "fixture": fx, "eval": ev}
        print(json.dumps(row), flush=True)
        with open(results_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        del model, over


def main():
    cmd = sys.argv[1]
    if cmd == "data":
        build_data()
    elif cmd == "ref":
        cmd_ref()
    elif cmd == "variant":
        cmd_variant(sys.argv[2:])
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
