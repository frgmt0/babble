"""Sampled-output sanity check: lean best-of-4 vs live HFGenerator, fixed seed.

Replies won't be token-identical (RNG consumption differs); they should be
sane and similar in character. Uses synthetic prompts only.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import bench.extreme.torch_lean.lean as lean  # noqa: E402
from bench.extreme.reference import MODEL_DIR  # noqa: E402

PROMPTS = [
    "user: hey booper whats up",
    "user: do you like cats or dogs",
    "user: whats your favorite game\nassistant: minecraft probably\nuser: why minecraft",
]


def main():
    torch.set_num_threads(4)
    from babble.config import Settings
    from babble.hfserve import HFGenerator

    s = Settings.from_env(ROOT)
    s = dataclasses.replace(s, hf_model_dir=MODEL_DIR, max_new_tokens=64, best_of=4,
                            temperature=0.5, top_k=40, top_p=0.9, repetition_penalty=1.15,
                            no_repeat_ngram_size=4)
    hf = HFGenerator(s)
    lean.W8BF_PREFILL_FP32 = True
    gen = lean.LeanGenerator(lean.LeanModel(mode="w8bf", head="rescore"), pad_id=hf.pad_id, eos_id=hf.eos_id)
    cfg = lean.SampleCfg(0.5, 40, 0.9, 1.15, 4)
    for p in PROMPTS:
        torch.manual_seed(1234)
        hf_text, _ = hf._generate(p, max_new_tokens=64, best_of=4)
        torch.manual_seed(1234)
        ids = hf._encode_prompt(p)[0].tolist()
        r = gen.generate(ids, n=4, max_new=64, cfg=cfg)
        keep = [t for t in r.tokens[r.best].tolist() if t not in (hf.pad_id, hf.eos_id)]
        print(f"PROMPT: {p!r}\n  HF   : {hf_text!r}\n  lean : {hf.tokenizer.decode(keep, skip_special_tokens=True).strip()!r}")
        for i in range(4):
            k = [t for t in r.tokens[i].tolist() if t not in (hf.pad_id, hf.eos_id)]
            print(f"    cand{i} lp={r.mean_logprob[i]:.3f}: {hf.tokenizer.decode(k, skip_special_tokens=True).strip()!r}")


if __name__ == "__main__":
    main()
