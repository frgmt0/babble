"""Decode benchmark for track w4 (int4 decode weights / two-stage head).

One process = one engine config (BABBLE_NATIVE_W4 / BABBLE_NATIVE_HEAD2 from
the env), on the live model. Run each under `flock /tmp/babble-bench.lock`,
interleaving configs; every metric is the median of REPS runs after a warmup.

* decode1:   1 stream, greedy, 128 new tokens after the ~27-token bench prompt,
             eos ignored -> tok/s over steps 2..128 (decode only)
* bo4:       best-of-4, live sampling (T .5, k 40, p .9, rep 1.15, ngram 4),
             64 new tokens each, eos ignored -> aggregate tok/s (whole call)
             and steady decode agg tok/s
* bo4_long:  same, with a ~1500-token 8-turn prompt (prefill dominates TTFT)

    flock /tmp/babble-bench.lock python bench/extreme/w4_bench.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL_DIR = Path(os.environ.get("W4_MODEL_DIR", "/home/jason/projects/babble/artifacts/hf-booper-longctx-v1"))
REPS = int(os.environ.get("W4_REPS", "7"))


def main():
    import torch
    from tokenizers import Tokenizer

    from babble.leanserve import SamplingConfig
    from babble.nativeserve import NativeEngine

    torch.set_num_threads(1)
    eng = NativeEngine(MODEL_DIR, threads=int(os.environ.get("NATIVE_THREADS", "4")))
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    bos, sep, eos = (tok.token_to_id(t) for t in ("<bos>", "<sep>", "<eos>"))
    short = [bos, *tok.encode("user: Write two sentences about a robot learning why people laugh.", add_special_tokens=False).ids, sep]
    hist = "\n".join(
        f"user: {u}\nassistant: {a}"
        for u, a in [
            ("hey booper whats up", "not much just vibing, been reading about space all day honestly"),
            ("did you see the game last night", "yeah that last goal was insane, nobody saw it coming"),
            ("i think the ref was kinda bad tho", "lol the ref is always bad, thats part of the fun"),
            ("what should i eat for dinner", "pizza is always the answer unless you had it yesterday"),
        ]
        * 6
    )
    long_ids = tok.encode(hist + "\nuser: anyway what are you up to this weekend", add_special_tokens=False).ids[-1500:]
    long = [bos, *long_ids, sep]
    live = SamplingConfig(temperature=0.5, top_k=40, top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=4)

    def run(ids, n, max_new, greedy, seed):
        o = eng.generate(ids, n=n, max_new=max_new, sampling=live, eos_id=eos, seed=seed, greedy=greedy, stop_at_eos=False)
        return o

    res = {}
    for name, ids, n, max_new, greedy in (
        ("decode1", short, 1, 128, True),
        ("bo4", short, 4, 64, False),
        ("bo4_long", long, 4, 64, False),
    ):
        run(ids, n, max_new, greedy, 0)  # warmup
        dec, agg, ttft = [], [], []
        for r in range(REPS):
            o = run(ids, n, max_new, greedy, 1 + r)
            dec.append(n * (max_new - 1) / (o.last_s - o.ttft_s))
            agg.append(n * max_new / o.total_s)
            ttft.append(o.ttft_s * 1e3)
        res[name] = {
            "prompt": len(ids),
            "decode_agg_tok_s": round(statistics.median(dec), 1),
            "e2e_agg_tok_s": round(statistics.median(agg), 1),
            "ttft_ms": round(statistics.median(ttft), 2),
        }
    out = {"w4": os.environ.get("BABBLE_NATIVE_W4", ""), "head2": os.environ.get("BABBLE_NATIVE_HEAD2", ""), **res}
    if eng.head2 is not None:
        out["head2_stats"] = eng.head2_stats()
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
