"""Benchmark the lean torch loop against HF generate (bench/extreme/README protocol).

Run each invocation under the shared lock:

    flock /tmp/babble-bench.lock python bench/extreme/torch_lean/bench.py \
        --backend lean --mode fp32 --threads 4

Prints one JSON line per metric. All prompts are synthetic.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from bench.extreme.reference import _LONG_HISTORY, MODEL_DIR  # noqa: E402
from bench.extreme.torch_lean.lean import LeanGenerator, LeanModel, SampleCfg  # noqa: E402

BENCH_PROMPT = "Write two sentences about a robot learning why people laugh."
CFG = SampleCfg(temperature=0.5, top_k=40, top_p=0.9, repetition_penalty=1.15, no_repeat_ngram_size=4)


def tok():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))


def prompt_ids(t, text: str) -> list[int]:
    return [t.token_to_id("<bos>"), *t.encode(text, add_special_tokens=False).ids, t.token_to_id("<sep>")]


def long_ids(t, n: int) -> list[int]:
    body = t.encode((_LONG_HISTORY + "\n") * 4, add_special_tokens=False).ids
    return [t.token_to_id("<bos>"), *body[-(n - 2) :], t.token_to_id("<sep>")]


def med(xs):
    return statistics.median(xs)


# ---------------------------------------------------------------- backends


class HFBackend:
    def __init__(self):
        from babble.hfserve import _load_int8

        self.model, _ = _load_int8(MODEL_DIR)

    @torch.inference_mode()
    def run(self, ids, *, n, max_new, greedy, ttft_only=False):
        x = torch.tensor([ids])
        kw = dict(
            max_new_tokens=1 if ttft_only else max_new,
            min_new_tokens=1 if ttft_only else max_new,  # EOS ignored
            num_return_sequences=n,
            pad_token_id=16380,
            eos_token_id=16383,
            use_cache=True,
        )
        if greedy:
            kw.update(do_sample=False)
        else:
            kw.update(
                do_sample=True,
                temperature=CFG.temperature,
                top_k=CFG.top_k,
                top_p=CFG.top_p,
                repetition_penalty=CFG.repetition_penalty,
                no_repeat_ngram_size=CFG.no_repeat_ngram_size,
            )
        t0 = time.perf_counter()
        self.model.generate(x, **kw)
        return time.perf_counter() - t0


class LeanBackend:
    def __init__(self, mode, head, compile_mode=None):
        self.model = LeanModel(mode=mode, head=head)
        step_fn = None
        self.compile_s = 0.0
        if compile_mode:
            from bench.extreme.torch_lean.compiled import make_step

            step_fn = make_step(self.model, compile_mode)
        self.gen = LeanGenerator(self.model, step_fn=step_fn)

    def run(self, ids, *, n, max_new, greedy, ttft_only=False, prefix_cache=None):
        r = self.gen.generate(
            ids,
            n=n,
            max_new=1 if ttft_only else max_new,
            cfg=CFG,
            greedy=greedy,
            ignore_eos=True,
            prefix_cache=prefix_cache,
        )
        return r.total_s


def timed(fn, runs):
    fn()  # warmup
    return [fn() for _ in range(runs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="lean", choices=["lean", "hf"])
    ap.add_argument("--mode", default="fp32")
    ap.add_argument("--head", default=None)
    ap.add_argument("--compile", default=None, help="compile mode for lean step (default/max-autotune-no-cudagraphs)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--what", default="decode1,bo4,ttft")
    ap.add_argument("--tag", default="")
    ap.add_argument("--prefill-fp32", action="store_true", help="w8bf: keep an fp32 copy for M>8 matmuls")
    ap.add_argument("--prefill-dequant", action="store_true", help="w8bf: dequantize per matmul when M>8")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.prefill_fp32:
        import bench.extreme.torch_lean.lean as lean_mod

        lean_mod.W8BF_PREFILL_FP32 = True
    if args.prefill_dequant:
        import bench.extreme.torch_lean.lean as lean_mod

        lean_mod.W8BF_PREFILL_DEQUANT = True
    torch.manual_seed(0)
    t = tok()
    t_load = time.perf_counter()
    be = HFBackend() if args.backend == "hf" else LeanBackend(args.mode, args.head, args.compile)
    load_s = time.perf_counter() - t_load
    label = dict(
        backend=args.backend,
        mode=args.mode if args.backend == "lean" else "hf-fp32",
        head=args.head,
        compile=args.compile,
        threads=args.threads,
        tag=args.tag,
        prefill_fp32=args.prefill_fp32,
        prefill_dequant=args.prefill_dequant,
        load_s=round(load_s, 2),
    )
    ids27 = prompt_ids(t, BENCH_PROMPT)
    what = args.what.split(",")

    def emit(**kv):
        print(json.dumps({**label, **kv}), flush=True)

    if "compilewarm" in what and args.backend == "lean" and args.compile:
        # First-call cost of the compiled step (includes inductor codegen).
        t0 = time.perf_counter()
        be.run(ids27, n=4, max_new=4, greedy=False)
        be.run(ids27, n=1, max_new=4, greedy=True)
        emit(metric="compile_first_calls_s", value=round(time.perf_counter() - t0, 2))

    if "decode1" in what:
        xs = timed(lambda: be.run(ids27, n=1, max_new=128, greedy=True), args.runs)
        emit(metric="decode1", prompt=len(ids27), new=128, wall_s=round(med(xs), 4), tok_s=round(128 / med(xs), 1))
    if "bo4" in what:
        xs = timed(lambda: be.run(ids27, n=4, max_new=64, greedy=False), args.runs)
        emit(
            metric="bestof4",
            prompt=len(ids27),
            new=64,
            wall_s=round(med(xs), 4),
            agg_tok_s=round(256 / med(xs), 1),
            per_stream_tok_s=round(64 / med(xs), 1),
        )
    if "ttft" in what:
        for L in (32, 128, 512):
            ids = long_ids(t, L)
            for n in (1, 4):
                xs = timed(lambda: be.run(ids, n=n, max_new=1, greedy=False, ttft_only=True), args.runs)
                emit(metric="ttft", prompt=len(ids), batch=n, ms=round(med(xs) * 1000, 1))
    if "prefix" in what and args.backend == "lean":
        # Multi-turn: 512-token prompt, all but the last 30 tokens already cached.
        ids = long_ids(t, 512)
        cut = len(ids) - 30
        m = be.model
        base = m.new_cache(1, len(ids) + 64)
        with torch.inference_mode():
            m.forward(torch.tensor([ids[:cut]]), base, 0)

        def one():
            # The generator writes past `cut` into the cache; positions < cut are
            # never touched, so the cached prefix stays valid across calls.
            return be.run(ids, n=4, max_new=1, greedy=False, ttft_only=True, prefix_cache=(base, cut))

        xs = timed(one, args.runs)
        emit(metric="ttft_prefix_cached", prompt=len(ids), cached=cut, batch=4, ms=round(med(xs) * 1000, 1))


if __name__ == "__main__":
    main()
