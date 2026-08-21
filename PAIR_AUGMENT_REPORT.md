# Correction-pair augmentation — with/without measurement, 2026-08-21

ro asked (booper channel): *"can we generate synthetic pairs? so like when i correct it we
create 3 possible variants that are semantically correct and then 50 turns into 150?"*

**tl;dr — build it, but don't turn it on.** The generator works: an LLM paraphraser
(`babble/llm.py` + `babble/pairaugment.py`) produces variants that stay register-matched and
semantically tied to the source pair, and the train/val split is provably leak-free (§2). But
the measurement (§3) does not show augmentation earning its keep at any N tested. The best cell
(1x) is a 0.007-nat improvement over baseline — smaller than that cell's own seed-to-seed noise
(std 0.006–0.016 across all cells). 3x and 5x are both *worse* than the no-augmentation baseline,
not better. Unlike the corpus-level synthetic generator (`PIPELINE_REVAMP_2026-08-20.md` §7.2),
the shuffled-word-order control here does **not** beat the real variants — real order comes out
ahead of shuffled at every N compared — so this is not the same "it's pure regularization"
story. It is instead a simpler, blunter finding: at this scale, augmenting correction pairs
moves val loss by less than noise, and adding more of it (3x, 5x) actively hurts. Recommend
**not** wiring this into booper's live post-train loop as currently measured; keep it
flag-gated and off by default, and re-run this grid against booper's real 50-pair corpus at
production model scale before reconsidering.

Everything below ran against fabricated stand-in data (`experiments/pairaugment_data.py`, 50
hand-written pairs, never real user corrections) and a tiny 2-layer/64-dim model
(`experiments/pairaugment_grid.py`), isolated under `experiments/pairaugment-scratch/`. Nothing
here touched `/home/beckett/babble-live` or booper's real `data/interactions.jsonl`, and nothing
was promoted.

## 1. What was built

- `babble/llm.py` — `LLMClient` interface + `ClaudeCLIClient` (shells out to `claude -p`),
  behind a config knob (`paraphrase_model`, default `haiku`).
- `babble/pairsplit.py` — `pair_split`, a hash-bucket train/val split for correction pairs,
  shared by the augmenter and by `posttrain.py` so both sides agree on which pairs are
  held out.
- `babble/pairaugment.py` — `generate_augmented_pairs` (the generator), `AugmentedPairStore`
  (its own file, `data/synthetic_pairs.jsonl` in production / `augmented_pairs.jsonl` in this
  run's scratch dir — never mutates `interactions.jsonl`), `assert_no_leakage` /
  `check_leakage` (the automated leak check), `register_comparison` (style-drift check),
  `AutoAugmentTrigger` (fires a background variant-generation call after a live correction).
- `posttrain.py` — val split now computed from real pairs only via `pairsplit.py`;
  `include_pair_augmentation` flag (`--augment-pairs` CLI / env knob) controls whether
  `trainable_augmented_pairs` is added to the training pool. Off by default.
- CLI: `babble augment-pairs` (generate + leakage check in one command), `babble
  augment-check` (leakage check only, for a file already on disk).
- Tests: `test_llm.py`, `test_pairsplit.py`, `test_pairaugment.py` — generator behavior,
  leakage check (including an injected-leak case that must fail loudly), register comparison.

## 2. Generation run + leakage check

`python -m experiments.pairaugment_generate --n 5 --workers 6` against the 50 stand-in pairs:

| | |
|---|---|
| Seeded pairs | 50 |
| Train-side / val-side (hash-bucket split) | 40 / 10 |
| Variants requested | up to 5 per train-side pair (200 ceiling) |
| Variants generated | **188** |
| Failed pairs | 0 |
| Blocked (blocklist) | 0 |

**Leakage check** (`assert_no_leakage`, also runnable standalone as `babble augment-check`):

```
188 checked, 188 train-side, 0 val-side, 0 leaked, 0 orphaned -- CLEAN
```

Every one of the 188 stored variants traces back to a train-side source pair id; none derive
from the 10 val-side pairs. This is the check the corpus-level generator was missing when
364/1200 of its rows carried val-only trigrams — here it's automated and it runs as part of
generation, not as an afterthought.

**Register comparison** (real pairs vs. generated variants, same run):

| | real (50) | variants (188) |
|---|---|---|
| mean chars | 20.9 | 21.1 |
| mean words | 4.0 | 4.1 |
| lowercase rate | 1.000 | 1.000 |
| punctuation rate | 0.020 | 0.016 |
| vocab size | 139 | 250 |
| vocab overlap | — | 0.480 |

`drifted=False` — length, casing, and punctuation match closely; vocabulary overlap of 0.48 is
expected (paraphrases introduce new words by construction) and not itself a drift signal.

Sample (source → variant), verbatim from `experiments/results/pairaugment/sample_variants.json`:

| source prompt | source chosen | variant prompt | variant chosen |
|---|---|---|---|
| `whats the holdup` | `waiting on approval still` | `where's it at` | `approval still pending` |
| `can you double check this` | `looks fine to me` | `double check this?` | `seems fine to me` |
| `whats your favorite snack` | `chips, always chips` | `best snack tbh` | `chips obviously` |

Meaning preserved, register preserved, no assistant-prose leakage ("Certainly! ..."). This part
of the ask is met.

## 3. The measurement: with/without, shuffled control, N sweep

`python -m experiments.pairaugment_grid` — tiny model (2 layer, 2 head, 64 embd, 128 block),
150 steps, 3 seeds per cell, same pretrained snapshot per seed reused across all cells at that
seed so augmentation pool is the only thing that varies. Checkpoint selection uses held-out
**real correction-pair val loss** (10 val-side pairs, never touched by any variant, any cell) —
exactly the number this feature is supposed to move.

| cell | augmented pairs added | seed 1 | seed 2 | seed 3 | **mean pair-val loss** | std |
|---|---|---|---|---|---|---|
| baseline-0x (no augmentation) | 0 | 3.0644 | 3.0861 | 3.0722 | **3.0742** | 0.0110 |
| aug-1x | 40 | 3.0662 | 3.0627 | 3.0741 | **3.0677** | 0.0058 |
| aug-3x | 120 | 3.0706 | 3.0783 | 3.0914 | **3.0801** | 0.0105 |
| aug-5x | 188 | 3.0744 | 3.0767 | 3.0954 | **3.0822** | 0.0115 |
| aug-3x-shuffled (control) | 120 | 3.0737 | 3.0843 | 3.1059 | **3.0880** | 0.0164 |

Lower is better (loss). Deltas vs. baseline:

| cell | Δ vs baseline |
|---|---|
| aug-1x | **−0.0066** (marginally better) |
| aug-3x | +0.0059 (worse) |
| aug-5x | +0.0079 (worse) |
| aug-3x-shuffled | +0.0137 (worst of all) |

### Shuffled control

The shuffled-word-order control (same 120 aug-3x variants, word order scrambled within prompt
and within chosen, same seed-1234 shuffle) scored **3.0880**, worse than the real aug-3x
variants it was built from (3.0801, a 0.0079-nat gap) and worse than baseline. This is the
**opposite** finding from the corpus-level synthetic generator, where the shuffled control won
outright and proved the gain there was pure token-count regularization. Here, real semantic
order beats scrambled nonsense at the same N — so if there is any signal at all in this data, it
is not purely "more tokens, no meaning."

That said: the real-vs-shuffled gap (0.0079) is smaller than the per-cell standard deviation in
either cell (0.0105 real, 0.0164 shuffled), and both are worse than the no-augmentation
baseline. So the honest statement is: **the shuffled control does not beat the real variants,
but neither result clears the noise floor.** Three seeds on a toy model is not enough to call
this significant either way — it just isn't the same failure mode as the corpus generator.

### N sweep

1x → 3x → 5x is **monotonically worse** as N increases (3.0677 → 3.0801 → 3.0822). More
augmented pairs did not help; it moved the model further from the real-pair val distribution,
consistent with dilution (each training step spends a larger fraction of its gradient on
paraphrased text and a smaller fraction on the 40 real pairs the val set is drawn from the same
population as) outweighing whatever regularization benefit extra tokens might otherwise buy on
a severely undertrained toy model. It stops helping immediately — there was no N tested where
augmentation clearly beat baseline outside noise.

## 4. Verdict

**Does not earn its keep, as measured.** Every augmented cell's mean is within about half a
standard deviation of baseline or worse; 3x and 5x are unambiguously worse than doing nothing;
and the one cell that edges out baseline (1x, −0.007 nats) does so by less than its own seed
noise. The generator itself is correctly built — leak-free by construction and measurement,
register-preserving, semantically sound by inspection (§2) — but "the pipeline is correct" and
"the pipeline helps" are different claims, and only the second one is what ro actually asked to
find out.

Recommendation:

- **Do not** flip `include_pair_augmentation` on by default for booper's live post-train loop.
  Keep it behind the flag it already has.
- If revisited, re-run this exact grid (`experiments/pairaugment_grid.py`) against booper's real
  ~50-pair correction set (not the stand-in data) and at a model scale closer to what's actually
  trained in production — a 2-layer/64-dim toy model with 150 steps is a plumbing smoke test,
  not a scale at which "more training data helps" is likely to show up cleanly either way.
- Do not scale N past 1x even experimentally without re-measuring; this run's data says more is
  actively worse, not just flat.

Everything this run touched lives under `experiments/pairaugment-scratch/` (gitignored data)
and `experiments/results/pairaugment/` (tracked summaries: `generate_summary.json`,
`sample_variants.json`, `grid.jsonl`). No changes were made to `/home/beckett/babble-live` or to
booper's real interaction data, and nothing was promoted.
