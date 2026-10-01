"""Long-context parity: the native engine vs transformers' fp32 forward.

    python bench/extreme/longctx_parity.py [--kv fp32,q16,fp16] [--model DIR] [--tokens 2048]

Runs one N-token sequence (repo docs as text, nothing from the corpus) through
the transformers fp32 forward of the int8 snapshot, then through the native
engine three ways per KV type:

* full prefill of all N tokens;
* a 1500-token prefill, then token-by-token decode to N (the decode path,
  split-K over the shared prompt + per-stream tail);
* an N/2 prefix snapshot exported, restored, and the rest prefilled (the
  multi-turn prefix-KV-reuse path).

Reports max |dlogit| (all positions and positions >= 1536), top-1 agreement
and the next-token NLL delta. The lossless bar is the same as reference.py:
top1 >= 0.995 and |dNLL| < 0.002.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", default="fp32,q16,fp16")
    ap.add_argument("--model", default=str(ROOT / "artifacts" / "hf-booper-longctx-v1"))
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    from babble.hfserve import _load_int8
    from babble.nativeserve import NativeEngine

    model_dir = Path(args.model)
    n = args.tokens
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    text = "\n".join(
        p.read_text() for p in (ROOT / "README.md", ROOT / "docs/reports/NATIVE_RUNTIME_2026-09-26.md", ROOT / "CLAUDE.md")
    )
    ids = [tok.token_to_id("<bos>"), *tok.encode(text, add_special_tokens=False).ids]
    if len(ids) < n:
        raise SystemExit(f"only {len(ids)} tokens of text for --tokens {n}")
    ids = ids[:n]

    torch.set_num_threads(args.threads)
    model, _ = _load_int8(model_dir)
    with torch.inference_mode():
        ref = model(torch.tensor([ids])).logits[0].float()
    del model
    tgt = torch.tensor(ids[1:])
    ref_nll = float(F.cross_entropy(ref[:-1], tgt))
    late = min(1536, n - 1)

    def report(name: str, got: torch.Tensor) -> None:
        d = float((got - ref).abs().max())
        d_late = float((got[late:] - ref[late:]).abs().max())
        top1 = float((got.argmax(-1) == ref.argmax(-1)).float().mean())
        dnll = float(F.cross_entropy(got[:-1], tgt)) - ref_nll
        ok = top1 >= 0.995 and abs(dnll) < 0.002
        print(
            f"  {name:30s} max|dlogit| {d:.3e} (pos>={late}: {d_late:.3e})  top1 {top1:.5f}  "
            f"dNLL {dnll:+.2e}  {'lossless' if ok else 'QUALITY TRADE'}",
            flush=True,
        )

    prefill = min(1500, n - 1)
    cut = n // 2
    for kv in args.kv.split(","):
        eng = NativeEngine(model_dir, threads=args.threads, kv=kv)
        print(f"kv={kv}  ({n} tokens, ref NLL {ref_nll:.5f})")
        report(f"full prefill {n}", eng.full_logits(ids))
        report(f"{prefill} prefill + {n - prefill} decode", eng.decode_logits(ids, prefill=prefill))
        head, snap = eng.forward(ids[:cut], export=True)
        tail, _ = eng.forward(ids, start=cut, kv_in=snap, kv_in_len=cut)
        report(f"{cut} snapshot + {n - cut} prefill", torch.cat([head, tail]))
        eng.close()


if __name__ == "__main__":
    main()
