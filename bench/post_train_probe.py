"""Fixed-probe sampler used for the corrections-only post-train A/B in
POST_TRAIN_EXPERIMENT.md.

Loads whatever checkpoint sits at `$BABBLE_CHECKPOINT_DIR/latest.pt` and runs a
fixed prompt set through it with seeded generators, so the *same* random draws
are replayed against every checkpoint it's pointed at -- the only thing that
can differ across runs (each pointed at a different `BABBLE_CHECKPOINT_DIR`) is
the model weights, not the RNG stream. Point it at an isolated checkpoint/data
copy via `BABBLE_CHECKPOINT_DIR` / `BABBLE_DATA_DIR` -- never at the live
install while `babble-bot` is running, to avoid racing its own checkpoint
writes.

    BABBLE_CHECKPOINT_DIR=... BABBLE_DATA_DIR=... python3 bench/post_train_probe.py <label>
"""
from __future__ import annotations

import hashlib
import json
import sys

import torch

from babble.config import Settings
from babble.generate import continue_text, load_model

PROBES = ["hola", "hello", "do you want to enter giveway", "why is", "the cat", "where"]
SAMPLES_PER_PROMPT = 3


def main() -> None:
    settings = Settings.from_env()
    model, step = load_model(settings)
    label = sys.argv[1] if len(sys.argv) > 1 else "run"

    out = {
        "label": label,
        "checkpoint_step": step,
        "params": model.num_params(),
        "temperature": settings.temperature,
        "top_k": settings.top_k,
        "max_new_tokens": settings.max_new_tokens,
        "probes": {},
    }

    for prompt in PROBES:
        # sha256, not the builtin hash(): PYTHONHASHSEED randomizes str hash()
        # per-process, which would silently desync the "same seed" guarantee
        # between two separate before/after invocations of this script.
        seed_base = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16) % (2**31)
        samples = []
        for i in range(SAMPLES_PER_PROMPT):
            gen = torch.Generator().manual_seed(seed_base + i)
            text = continue_text(
                model,
                prompt,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_k=settings.top_k,
                generator=gen,
            )
            samples.append(text)
        out["probes"][prompt] = samples

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
