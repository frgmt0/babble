"""Gate entry points for track w4 against the *live* model (longctx-v1).

The repo fixture (`artifacts/perf-ref/ref.pt`) was built from multiturn-v1;
`w4_quant_study.py ref` rebuilt the same 5 fixture cases from longctx-v1 into
$W4_DIR/ref-longctx.pt. These functions run the native engine on longctx-v1
with whatever `BABBLE_NATIVE_W4` / `BABBLE_NATIVE_HEAD2` is set in the env.

    python bench/extreme/w4_check.py decode|full|prefix
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import bench.extreme.reference as ref  # noqa: E402

MODEL_DIR = Path(os.environ.get("W4_MODEL_DIR", "/home/jason/projects/babble/artifacts/hf-booper-longctx-v1"))
ref.REF_PATH = Path(os.environ.get("W4_DIR", "/tmp/maxperf/w4")) / "ref-longctx.pt"
_ENG = None


def engine():
    global _ENG
    if _ENG is None:
        from babble.nativeserve import NativeEngine

        _ENG = NativeEngine(MODEL_DIR, threads=int(os.environ.get("NATIVE_THREADS", "4")))
    return _ENG


def full(ids):
    return engine().full_logits(ids)


def decode(ids):
    return engine().decode_logits(ids, prefill=1)


def samplecmp(w4: str = "", head2: str = "", n_prompts: int = 120, freq: float = 0.0) -> dict:
    """Live-shape best-of-4 sampling, exact engine vs candidate engine, same seeds.

    Prompts: the w4 eval set's prompts (real-looking chat, up to 8 turns), each
    `<bos> prompt <sep>`. If the candidate's post-warp distribution equals the
    exact one, every sampled token and every candidate's summed logprob match.
    """
    import torch

    from babble.leanserve import SamplingConfig
    from babble.nativeserve import NativeEngine

    th = int(os.environ.get("NATIVE_THREADS", "4"))
    exact = NativeEngine(MODEL_DIR, threads=th, w4="", head2="")
    cand = NativeEngine(MODEL_DIR, threads=th, w4=w4, head2=head2)
    data = torch.load(Path(os.environ.get("W4_DIR", "/tmp/maxperf/w4")) / "evalset.pt")["eval"]
    step = max(1, len(data) // n_prompts)
    cfg = SamplingConfig(temperature=0.5, top_k=40, top_p=0.9, repetition_penalty=1.15,
                         no_repeat_ngram_size=4, frequency_penalty=freq)
    eos = 16383
    same_tok = same_streams = streams = tokens = 0
    max_dlp = 0.0
    best_same = 0
    for i, s in enumerate(data[::step][:n_prompts]):
        ids = s["ids"][: s["sep_at"] + 1]
        a = exact.generate(ids, n=4, max_new=64, sampling=cfg, eos_id=eos, seed=1000 + i)
        b = cand.generate(ids, n=4, max_new=64, sampling=cfg, eos_id=eos, seed=1000 + i)
        best_same += int(a.best == b.best)
        for ta, tb, la, lb in zip(a.tokens, b.tokens, a.mean_logprob, b.mean_logprob):
            streams += 1
            same_streams += int(list(ta) == list(tb))
            k = 0
            while k < min(len(ta), len(tb)) and ta[k] == tb[k]:
                k += 1
            same_tok += k
            tokens += len(ta)
            if list(ta) == list(tb):
                max_dlp = max(max_dlp, abs(la - lb))
    out = {"w4": w4, "head2": head2, "freq": freq, "prompts": n_prompts, "streams": streams,
           "identical_streams": same_streams / streams, "identical_prefix_tokens": same_tok / tokens,
           "tokens": tokens, "same_best": best_same / n_prompts, "max_dmean_logprob_identical": max_dlp}
    if head2:
        out["head2_stats"] = cand.head2_stats()
    return out


def evalnll(w4: str = "") -> dict:
    """Live-shape quality on the w4 eval set: prompt prefilled (int8), response decoded
    token by token through the engine (int4 copies when `w4` is set). Response-token
    NLL and top-1 vs the int8 fp32 reference stats from `w4_quant_study.py ref`."""
    import torch

    from babble.nativeserve import NativeEngine

    W = Path(os.environ.get("W4_DIR", "/tmp/maxperf/w4"))
    eng = NativeEngine(MODEL_DIR, threads=int(os.environ.get("NATIVE_THREADS", "4")), w4=w4, head2="")
    data = torch.load(W / "evalset.pt")["eval"]
    ref_stats = torch.load(W / "ref-eval-stats.pt")
    out = {}
    tot = {"all": [0.0, 0.0, 0, 0]}
    for s, r in zip(data, ref_stats):
        ids, a = s["ids"], s["sep_at"]
        lg = eng.decode_logits(ids, prefill=a + 1)[a:-1]
        tgt = torch.tensor(ids[a + 1 :])
        lp = lg.log_softmax(-1).gather(1, tgt.unsqueeze(1)).squeeze(1)
        agree = int((lg.argmax(-1) == r["argmax"][a:-1]).sum())
        for k in ("all", s["src"]):
            t = tot.setdefault(k, [0.0, 0.0, 0, 0])
            t[0] += -float(r["tlp"][a:].sum())
            t[1] += -float(lp.sum())
            t[2] += len(tgt)
            t[3] += agree
    for k, (rn, cn, n, ag) in tot.items():
        out[f"dnll_{k}"] = (cn - rn) / n
        out[f"top1_{k}"] = ag / n
        out[f"tokens_{k}"] = n
    return {"w4": w4, **out}


if __name__ == "__main__":
    if sys.argv[1] == "evalnll":
        print(evalnll(w4=os.environ.get("BABBLE_NATIVE_W4", "")))
    elif sys.argv[1] == "samplecmp":
        print(samplecmp(w4=os.environ.get("BABBLE_NATIVE_W4", ""), head2=os.environ.get("BABBLE_NATIVE_HEAD2", ""),
                        freq=float(os.environ.get("W4_FREQ", "0"))))
    else:
        print(os.environ.get("BABBLE_NATIVE_W4", ""), os.environ.get("BABBLE_NATIVE_HEAD2", ""), sys.argv[1],
              ref.compare(globals()[sys.argv[1]]))
