# Pipeline revamp + corpus-internal synthetic expansion — 2026-08-20

ro asked for two things: make the training pipeline actually work, and build
synthetic data that "mimics yet expands the pairs" using only the corpus the
model knows. This is the evidence report for both. Hard constraint honoured
throughout: **no external text, no pretrained weights** — every byte the model
sees is the collected Discord corpus (407 rows, ~25 KB, median row 25 bytes,
4 authors) or a labelled recombination of it.

## TL;DR

- The early stop was firing inside the measurement noise. Fixed with a
  min-steps floor and a noise-band stall margin, both measured.
- Val loss on held-out rows genuinely turns at ~250–500 steps; everything
  after that is memorisation of the train split. "Train to convergence" and
  "generalise to unseen rows" are *different targets* on this corpus, and the
  gap between them is the honest headline: 25 KB of chat does not train a
  general model at any setting we swept.
- The served checkpoint (post-trained latest.pt) was measurably worse than
  the pretrain it started from: corpus val 3.91 vs 2.76. Post-train now runs
  at its own low LR with corpus rehearsal, refuses to run below a pair floor,
  and cannot promote a checkpoint that regresses held-out corpus val.
- Synthetic expansion measurably helps pretraining: −0.07 nats of held-out
  real-corpus val on the winner config, replicated 3 seeds vs 3 seeds
  (2.581 vs 2.647) — about a quarter of what a leaky first measurement
  claimed. But the *shuffle control* — same synthetic rows with word order
  destroyed, on the old untuned recipe — posts the best number in the whole
  sweep (2.5500). The gain is dilution/regularisation, not voice mimicry
  (§7.2, Finding 2).
- New defaults landed (lr 3e-4, dropout 0.2, cosine, 3x synthetic,
  1600-step budget), all env-flippable (§8). An interpolated byte trigram
  (2.23) still beats every transformer config swept (§2.5, §9).

## 1. The early stop was firing on noise — quantified

Method: hold `pretrained.pt` fixed, recompute val loss over 4,000 random
81-row holdouts of the same 407-row corpus (`experiments/val_noise.py`).
Per-row losses are computed once; each split's val loss is the token-weighted
mean over its rows, so this isolates *split* noise exactly.

```
splits 4000, holdout 81 rows
mean 2.6269   std 0.0541
p5   2.5433   p25 2.5894   median 2.6247   p75 2.6631   p95 2.7207
IQR 0.0737    p95-p5 0.1774
```

The wobble that killed the live run at step 350 was **0.075** (2.751 →
2.826). The split-noise IQR alone is 0.074. A patience stop reacting to a
0.075 move on an 81-row val set is reacting to which rows are in the split,
not to the model. Two more facts worth keeping:

- The trainer's fixed hash-split scores 2.761 against a random-split mean of
  2.627 — the live split happens to sit at the ~97th percentile of hardness.
  Nothing wrong with that (it is stable, which matters more), but live val
  numbers should never be compared against numbers from other splits.
- This is *between-split* spread. Within a run the split is fixed, but the
  checkpoint-to-checkpoint val wobble in the flat region of the long runs
  below is the same order of magnitude (see §2's curve).

**The fix** (landed, config-flippable):

- `BABBLE_TRAIN_STEPS` default 400 → **1000**, `BABBLE_TRAIN_MIN_STEPS`
  **600**: patience cannot fire before the floor, and the floor sits inside
  the budget by construction (a floor outside the budget makes patience dead
  code — an earlier draft of this fix had exactly that bug). 600 is set from
  the measured curves: every configuration swept has its val minimum well
  before step 600, so a stop after the floor is a stop after the turn, never
  before it.
- `BABBLE_TRAIN_STALL_MARGIN` (default 0.05 ≈ 1σ of measured noise): a
  checkpoint burns patience only when val exceeds the best seen by more than
  the margin. Movement inside the band is neutral — neither a new best nor a
  stall. Margin 0 restores the old behaviour.
- Best-val checkpoint selection is unchanged: extra steps cost time, never
  quality, because the winning write is still the best-val state.

A subtlety the noise study surfaced: the *split*-noise std (0.054) is
common-mode within a run — the split is fixed, so it cannot mis-rank two
checkpoints of the same run. What it does poison is (a) any comparison of
val numbers across different splits/dates, and (b) hair-trigger stopping,
because the checkpoint-to-checkpoint wobble on the fixed split (optimizer
noise) is the same order of magnitude — the long-run curves show ±0.03–0.06
between adjacent checkpoints in flat regions. The margin covers both.

Not adopted: k-fold/repeated-split val (k models per decision is k× the
train cost on a box that also serves the bot; the margin gets the same
robustness for free) and EMA smoothing (equivalent role to the margin, one
more tunable, harder to explain in a log line). Patience itself is nearly
moot at these budgets — a full 1000-step run is minutes — but it is kept,
fixed, for anyone who raises the ceiling.

## 2. Convergence: where train bottoms and where val turns

Runs with early stop disabled (`experiments/sweep.py`, seed 1, same split as
live): default 3.3M model, lr 1e-3, batch 8.

```
step   250: train_full 2.38  val 2.79   <- val minimum, and it never returns
step   500: train_full 1.99  val 3.04
step   750: train_full 1.58  val 3.52
step  1000: train_full 1.29  val 3.77
step  1500: train_full 0.82  val 4.45
step  2000: train_full 0.49  val 5.27
step  2500: train_full 0.34  val 5.82
step  3000: train_full 0.29  val 6.16
step  3250: train_full 0.26  val 6.36   <- run stopped here, deliberately
```

This run was budgeted for 50k steps and **stopped on purpose at step 3250**
(ro's call, and the right one): the question it existed to answer — "does
val genuinely turn, or was the early stop firing on split noise?" — was
already answered beyond ambiguity. Val bottomed at 2.794 by step 250 and
then climbed *monotonically* for 3,000 straight steps to 6.36 while train
loss fell to 0.26 (near-perfect memorisation of the 326 train rows). That
is real overfitting, not noise; ten more hours of curve would have bought
nothing and the cores were needed by the decision lanes. The 2k/10k/50k
prefix framing collapses to: 2k is already deep in the memorisation regime
(val 5.27), and 10k/50k could only continue a curve that had been monotone
for 3,000 steps.

Companion evidence from the completed 2400- and 4000-step runs in §§3–4:
every non-synthetic config has train loss heading for ~0.2–0.5 (near-perfect
memorisation of 326 rows) by 2.4k–4k steps while val climbs past 4–6.

Val turns just past ~250 steps and never comes back; train keeps falling
toward memorisation. The old 400-step budget was not "undertrained" for
*val* — it was undertrained for *train-loss floor / memorisation*, which is
a different objective with a different defensible use (parroting corpus
lines in-voice vs babbling in-distribution). §6 shows probe samples from
both regimes so the choice is a decision, not a vibe.

The original diagnosis ("killed 350 steps in while still learning") was
therefore half right: the stop *was* firing on noise (§1), but longer
training at the default recipe only memorises harder. What actually moves
held-out val is the data (§7), not the step budget.

## 2.5 The number that reframes everything: a trigram beats the transformer

An interpolated byte n-gram (`experiments/ngram_baseline.py`) fit on the
same 326 train rows and scored on the same 81 held-out rows with the same
token accounting:

| model | val (nats/byte) |
|---|---|
| unigram | 3.01 |
| bigram (interpolated) | 2.50 |
| **trigram (interpolated)** | **2.23** |
| served `pretrained.pt` (3.3M transformer, step 200) | 2.76 |

A count-based trigram that fits in milliseconds beats the shipped 3.3M-param
transformer by **~0.5 nats/byte** on held-out corpus text. (An independent
implementation during design review, with per-context backoff, got 2.10 —
same conclusion, stronger.) This kills the "corpus is too small" explanation
for the current quality: 25 KB is enough to do *much* better than what was
being served. The binding constraint was the recipe. The n-gram number now
stands as the acceptance floor any config change is judged against.

## 3. LR sweep (long budgets)

All runs: 3.3M default shape, batch 8, fixed hash split (326 train / 81
val), stop disabled, best-val checkpoint kept (`experiments/sweep.py`).
Remember the yardsticks: split-noise IQR 0.074, and the trigram at 2.23.

| lr | schedule | best val @ step | val @ 2k | final train | steps |
|---|---|---|---|---|---|
| 3e-4 | const | **2.650** @ 200 | 4.90 | 0.22 | 2400 |
| 3e-4 + dropout 0.2 | cosine, 3 seeds | **2.618 / 2.639 / 2.634** @ 300–350 | — | 1.80 | 1000 |
| 1e-3 (live default) | const | 2.794 @ 250 | 5.27 | 0.26 @ 3.25k | stopped @ 3250 (§2) |
| 1e-3 | cosine | 2.742 @ 150 | — | 0.94 | 1000 |
| 3e-3 | const | 2.768 @ 2200 | 2.79 | 2.69 | 2400 |

Reading it honestly:

- **3e-4 beats 1e-3 by ~0.14–0.18 nats** — bigger than split noise, and it
  replicates across three seeds (spread 0.021). ro's original hypothesis was
  right in direction, wrong in mechanism: lower LR doesn't fix the early
  stop, it finds a slightly better val minimum before the same memorisation
  cliff.
- **3e-3 never converges**: train loss hovers at 2.7 forever. Its "best val
  @ 2200" is not learning, it's a random walk grazing the noise band. Too
  hot at this batch size.
- Every constant-LR config still memorises by 2k steps (val ≥ 4.9). The LR
  moves the minimum a tenth of a nat; it does not change the shape of the
  curve.

## 4. Capacity / dropout / weight decay

Same protocol as §3; every cell compared against the properly-trained
lr 1e-3 baseline (2.794), never the prematurely-stopped live run.

| variation | params | best val @ step | Δ vs 2.794 baseline |
|---|---|---|---|
| 1M (3L·160d) | 1.05M | 2.679 @ 200 | −0.12 |
| 400k (2L·112d) | 0.39M | 2.702 @ 200 | −0.09 |
| 150k (2L·64d) | 0.15M | 2.705 @ 400 | −0.09 |
| dropout 0.1 | 3.35M | 2.712 @ 200 | −0.08 |
| dropout 0.2 | 3.35M | 2.731 @ 200 | −0.06 |
| dropout 0.3 | 3.35M | 2.702 @ 200 | −0.09 |
| wd 0 | 3.35M | 2.756 @ 200 | −0.04 |
| wd 0.1 | 3.35M | 2.746 @ 200 | −0.05 |
| small+drop combo (3L·128d, drop 0.1, wd 0.05, cosine), 3 seeds | 0.69M | 2.671 / 2.679 / 2.691 | −0.11 |

Reading it honestly:

- Every single-lever delta here is **0.04–0.12 nats — at or inside the
  split-noise IQR (0.074)**, single-seeded. Direction is consistently
  "slightly better than the untuned baseline", but no individual lever is a
  finding. The 20× params reduction costing ~nothing *is* a finding of
  sorts: capacity was never the binding constraint, in either direction.
- The two multi-seed combos (§3's 3e-4+drop0.2 at 2.63, small+drop at 2.68)
  are real but small improvements: −0.16 and −0.11 replicated.
- Nothing in this table approaches the trigram (2.23) or the synthetic mix
  (§7). Architecture and regularisation knobs are the wrong aisle for this
  corpus.

## 5. Post-train: measured harm, and the guardrails that stop it

Direct comparison of the two live checkpoints on the trainer's own held-out
split (`experiments/probes.py`, temp 0.5, fixed seed):

```
pretrained.pt (step 200):  corpus val 2.7613, corpus train 2.5797
latest.pt     (step  50):  corpus val 3.9081, corpus train 3.6700  <- SERVED
```

The bot is serving a checkpoint that is worse than its own starting point by
+1.15 nats on held-out corpus text — and the samples agree (§6). Mechanism,
in two parts:

1. **38 pairs at lr 1e-3 is weight demolition.** Pair-val rose from its very
   first checkpoint (3.59 → 5.56 over 200 steps) while pair-train memorised
   (2.32 → 0.32). The "best-val" checkpoint kept was simply the least-bad
   point of a run that only ever got worse.
2. **The bot never uses what post-train teaches.** Serving generates plain
   continuations (`best_continuation`, `<bos> text` layout). Post-train
   optimises the `<bos> prompt <sep> response` layout. The `<sep>` token
   never appears at inference, so the *only* effect post-train can have on
   the served bot is collateral damage to the continuation ability it
   actually uses.

**Fixes landed** (each measured in `experiments/post_grid.py`, table below):

- `BABBLE_POST_LEARNING_RATE` (default 1e-4): post-train no longer borrows
  the pretrain LR.
- `BABBLE_POST_REHEARSAL` (default 0.5): half of every post-train batch is
  plain corpus text under the pretrain objective, so the fine-tune cannot
  drift off the corpus distribution unopposed.
- `BABBLE_POST_MIN_PAIRS` (default 100): below this many trainable pairs the
  run refuses to start (`--force` overrides for experiments).
- **Promotion gate** (`BABBLE_POST_GATE_MARGIN`, default 0.05): after
  training, the candidate is scored against the pretrain snapshot on the
  held-out corpus split. Worse by more than the margin → `latest.pt` is not
  written, the run reports itself `gated`. Nothing mid-run writes
  `latest.pt` any more, so a half-finished fine-tune can never ship either.

Measured grids (`experiments/post_grid.py`, 200-step budget, 47 pairs, gate
margin 0.05). First under the **historical pair layout**, reproducing the
live failure exactly — the `old-lr1e-3` cell lands within 0.001 of the served
checkpoint's 3.9081:

| cell (pair layout) | lr | corpus val before → after | pair val | gate verdict |
|---|---|---|---|---|
| old-lr1e-3 (the live recipe) | 1e-3 | 2.761 → **3.909** | 3.597 | **blocked** |
| lr1e-4 | 1e-4 | 2.761 → 2.875 | 2.865 | blocked |
| lr1e-5 | 1e-5 | 2.761 → 2.760 | 2.821 | promoted |

(The remaining pair-layout cells died with the first measurement session;
they aren't needed — the layout itself is retired below. The full 9-cell
grid under the continuation layout follows.)

The full 9-cell grid under the **continuation layout** (now the default:
post-train examples are `<bos> prompt-text response-text` plain
continuations, matching what serving actually does — the `<sep>` mismatch
in mechanism 2 is retired). `synth` = the 643 synthetic pairs (504
continuation-cut + 139 postulated-prompt) mixed in; gate margin 0.05; 47
human pairs:

| cell | lr | rehearsal | synth pairs | corpus val 2.761 → | gate verdict |
|---|---|---|---|---|---|
| old-lr1e-3 | 1e-3 | — | — | 3.821 | **blocked** |
| lr1e-3+rehearse | 1e-3 | 0.5 | — | 2.855 | blocked |
| lr1e-4 | 1e-4 | — | — | 2.879 | blocked |
| lr1e-4+rehearse | 1e-4 | 0.5 | — | 2.748 | promoted |
| lr1e-5 | 1e-5 | — | — | 2.744 | promoted |
| lr1e-5+rehearse | 1e-5 | 0.5 | — | 2.729 | promoted |
| **lr1e-4+synth** | 1e-4 | — | 643 | **2.651** | **promoted** |
| lr1e-4+rehearse+synth | 1e-4 | 0.5 | 643 | 2.666 | promoted |
| lr1e-5+rehearse+synth | 1e-5 | 0.5 | 643 | 2.709 | promoted |

Three things this grid establishes:

1. **The gate works as specced**: every cell that damages corpus val by more
   than the margin is refused; the live-recipe cell (3.821) can never ship
   again.
2. **The old recipe was the worst possible cell** — highest LR, no
   rehearsal, no expansion, wrong layout.
3. The best post-train (**lr 1e-4 + synthetic pairs**) now
   *improves* held-out corpus val, 2.761 → 2.651. With the response pool
   actually expanded (§7), post-train stops being a tax and becomes a second
   pass of in-distribution training. −0.11 is at the edge of the noise band,
   so the claim is "no longer harmful, probably mildly helpful", not
   "solved" — but the *sign flip* from +1.15 of damage is the point.

## 6. Probe samples

Fixed prompts, fixed seed, temp 0.5, two samples per probe
(`experiments/probes.py`). First the two checkpoints the bot has actually
had, answering "is post-train helping" directly:

**`pretrained.pt`** (step 200, corpus val 2.761):

```
hello   → " hit s wot it t in at oning min me s is is pe ike the t arasse u"
the cat → " his s whe it t in at we mat in mene is is pe ing the t itats is"
why is  → " his s wounit t in at out fou w me we s is pe the the t atathe u"
boop    → " he"
```

**`latest.pt`** — post-trained, **what the bot serves** (corpus val 3.908):

```
hello   → "llyo o i your y y y oror"
the cat → " he"                          (second sample: " :e")
why is  → " ay f-be y jerer onatars manflany myet butipyeustutinye"
boop    → " s fe"
```

Post-train is plainly hurting: the pretrain babbles word-shaped
pseudo-English; the served checkpoint has lost even that. The numbers and the
samples agree.

**`b-drop0.2-lr3e-4-s1-best.pt`** — the best non-synthetic config from §3
(corpus val 2.618):

```
hello   → " so"                           (second sample: "")
the cat → " s afou lyou an in an onoman in t as as in pe int tee t arass ta"
why is  → " sis s ilyoura it the is mat in me s as is pe therese t itasst a"
boop    → ""                              (second sample: "")
```

Honest read: −0.18 nats of val loss does **not** buy visibly better text.
The tuned baseline babbles the same pseudo-English as the old pretrain —
fragments of "you/the/in/as" glued with plausible letter transitions. At
this corpus size the difference between 2.79 and 2.62 is statistical, not
legible.

**`win-synth-s1`** — winner config + 3x synthetic, best sweep candidate
(corpus val 2.571; end-of-run samples from the sweep harness):

```
hello   → " youh is re that ats e fonthe ou cathind jus the ats me is o ind"
the cat → "s the the"
why is  → " thats the ithat fue ithe nd touroror orou ack s ing be in forou"
boop    → "er abe is"
```

**`synth3x-shuf`** — the shuffle control (corpus val 2.550):

```
hello   → " to the can that pos won going gh the though the you the the thi"
the cat → "s the what yeats whave then docks the thou thats that he thot be"
why is  → " thats the whe the would think go the couse can the whe think th"
boop    → "er an what yeve the ould to do good ther we clos to whe to to th"
```

Worth noticing: the control trained on word-salad produces the most
word-like samples of the sweep — more real corpus words ("can", "going",
"think", "would", "good") and fewer glued fragments than the candidate
trained on carefully-ordered recombinations. Consistent with §7.2's
mechanism story: what these models learn from synthetic text is words and
word-shapes, not order.

### 6.1 The final model: before and after, one table

The final model was trained 2026-08-21 03:39–03:52 UTC on a snapshot of the
live corpus (415 rows, 332 train / 83 val) with the shipped defaults from §8:
lr 3e-4, dropout 0.2, cosine over 1600 steps, 3x train-only synthetic mix.
Best checkpoint at step 850, corpus val 2.5627. Then the *gated* post-train
(47 human pairs + 659 synthetic pairs, lr 1e-4, rehearsal 0.5, continuation
layout) ran on top and the gate **promoted**: corpus val 2.5627 → 2.4860.
That is the first post-train in this project's history measured to leave the
model *better* on held-out real corpus than its own starting point — the
same stage that used to cost +1.15 nats now buys −0.077.

Everything below is one probe run (`experiments/probes.py`, seed 7, temp
0.5), all four checkpoints scored on the identical 83-row held-out split:

| checkpoint | corpus val | what it is |
|---|---|---|
| old `latest.pt` | 3.903 | **what booper served before this run** — old post-train damage |
| old `pretrained.pt` | 2.757 | the undertrained 350-step pretrain under it |
| final `pretrained.pt` (step 850) | 2.563 | new recipe, trained to the real val turn |
| final `latest.pt` (post step 150) | **2.486** | **what booper serves now** |
| interpolated trigram | 2.23 | still the best likelihood model in the building |

Old served checkpoint:

```
hello   → "llyo o i your y y y oror"
the cat → " he"                          (second sample: " :e")
why is  → " ay f-be y jerer onatars manflany myet butipyeustutinye"
boop    → " s fe"
```

New served checkpoint (final `latest.pt`):

```
hello   → " he"        (second sample: " who that is you w dadour thathouryou mat o d inthe athe won and")
the cat → "s the the whe thatheratsthere d theskss ts pr therke ffistatst t"
why is  → " thats the ithe it as workerke th as as is pe the the thathest s"
boop    → "es if"      (second sample: "ere it it as at an bang ine sore orothe th inouthat and ow t ane")
```

Honest read of the 1.42-nat improvement in the served number: roughly 1.15
of it is *undoing the old post-train's damage* (3.90 → 2.76 was free — stop
shipping the harmful checkpoint), ~0.19 is the retuned pretrain recipe, and
−0.077 is the newly-non-harmful post-train. The text is still pseudo-English
babble — more word-shaped than before ("thats the", "who that is you"), but
nobody would mistake it for a sentence, and the trigram at 2.23 still beats
everything with weights. The bot is better because it stopped being actively
damaged and because training now stops where val actually bottoms, not
because 415 rows learned to talk.

## 7. Synthetic expansion — what was built and what it measured

PR #17's generator only synthesized the *prompt* side of pairs; the response
pool never grew, which is why it measured as a null result. Two new
generators, both strictly corpus-internal, both labelled and switchable:

- **`babble synth-corpus`** (`data/synthetic_corpus.jsonl`): order-2
  word-level Markov recombination of corpus phrasing. Every word is a word
  someone typed, spelled as they typed it; every transition follows an
  observed trigram (order-1 backoff only where no trigram continues).
  Verbatim replays of real rows are rejected, so every stored row is *new*
  text in the corpus voice. Mixed into the **train side only** via
  `BABBLE_TRAIN_SYNTHETIC=1`; val stays 100% real held-out rows.
- **Continuation-cut pairs** (`babble synth-generate`, same
  `synthetic_pairs.jsonl`, method-tagged `continuation_cut`): each real row
  ≥4 words is cut at word boundaries into (prefix → rest) pairs. Both halves
  verbatim corpus text — this is the pair-side expansion where the
  *response* pool finally grows. 407 rows → 504 pairs, vs 47 human pairs.
- **Rejected on evidence: adjacency pairing.** 340/403 chronologically
  adjacent corpus pairs are same-author, and the cross-author remainder are
  non-sequiturs from a bot-poke channel. There is no real dialogue structure
  in this corpus to mine.

### 7.1 First result, and the leak that made us distrust it

The first sweep mixed Markov rows into pretrain at two scales, and the 3x
mix was the best number in the entire sweep — the only lever that changed
the *shape* of the curve rather than nudging its minimum:

| run | synthetic rows | best val @ step | val @ 2k |
|---|---|---|---|
| no synthetic (lr 1e-3 baseline) | 0 | 2.794 @ 250 | ~4.9 |
| synth-1x | 400 | 2.689 @ 1400 | 2.82 |
| synth-3x | 1200 | **2.468 @ 2400** | 2.49 |

But before trusting it we audited the generator, and it had a leak: the
Markov chain was built over the **whole** consented corpus, val rows
included. No synthetic row ever enters val (the val side is real rows only,
by construction in both the trainer and `experiments/sweep.py`), but the
chain could splice *held-out phrasing* into training text. Measured, it
did: of the 81 val rows' 1,029 word trigrams, 950 appear in no train row —
and **364 of the 1,200 synthetic rows (30%) contained at least one of them,
1,744 splices in all**. Training had partial access to val phrasing, so
2.468 flatters the model by an unknown amount.

Fix landed: `generate_synthetic_corpus` now excludes val-side rows from the
chain by default (`babble/valsplit.py` is the split's single torch-free
definition, shared with the trainer; `--include-val-sources` restores the
old behaviour for experiments). The regenerated train-only 3x corpus
contains **zero** val-only trigrams, verified.

### 7.2 The decision-grade rerun: train-only sources, 3 seeds, controls

Reruns on the leak-free corpus, same protocol, plus two controls aimed at
the obvious confound — "is 3x better because recombination carries signal,
or would any 3× pile of in-vocabulary text delay memorisation the same
way?":

- **Word-shuffle control**: the same 1,200 rows with word order destroyed
  within each row (same size, same vocabulary, same per-row unigram counts,
  no corpus word order). If dilution/regularisation is the whole story,
  this should match the real mix.
- **5x scale test**: 2,000 rows, does more keep helping?

On the token-matched confound specifically: batch size and example length
distributions are identical across these runs, so at any given step every
run has seen the same number of training tokens — the no-synthetic baseline
at step 2400 (same tokens seen as the 3x run's best step) is at val ~4.9,
deep into memorisation. "More tokens before the val split gets memorised"
is not a confound *against* the synthetic mix; it is the mechanism's name.
The live question the shuffle control answers is whether the tokens must be
corpus-*ordered* to help.

The results, every run on the leak-free train-only synthetic corpus, same
fixed hash split (326 train / 81 val real rows), best-val checkpoint:

| run | config | synthetic | seeds | best val |
|---|---|---|---|---|
| win-nosynth | lr 3e-4, drop 0.2, cosine 1600 | none | s2/s3 (s1 truncated¹) | 2.6464 / 2.6474 → **mean 2.647** |
| win-synth | lr 3e-4, drop 0.2, cosine 1600 | 3x, real order | 3 | 2.5706 / 2.5996 / 2.5731 → **mean 2.581** |
| synth3xT | old recipe (lr 1e-3, no drop, const) | 3x, real order | 3 | 2.7047 / 2.7104 / 2.7020 → **mean 2.706** |
| **synth3x-shuf** | old recipe (lr 1e-3, no drop, const) | 3x, **word-shuffled** | 1 | **2.5500** @ 1200 |
| synth5xT | old recipe | 5x, real order | 1 (truncated¹) | 2.687 @ 200, rising monotonically after |
| (baseline, §2) | old recipe | none | 1 | 2.794 |

¹ win-nosynth-s1 and synth5xT-s1 were terminated at step 1400 when the
machine was reclaimed; their best-so-far values (2.6548 @ 200 and 2.687 @
200, both with val already climbing) are consistent with their lanes and
are not used in the means.

**Finding 1 — synthetic helps, by ~0.07, replicated.** On the winner config
the ± synthetic gap is 2.581 vs 2.647 across seeds whose spread (0.029 and
0.001) is well under the gap. That is a real effect — and it is roughly a
*quarter* of the 0.33 the leaky first sweep claimed (2.468 vs 2.794). The
first number was wrong because the Markov chain had seen val-side phrasing
(§7.1); this one is clean. Both numbers are stated here on purpose: the
leak was worth three times the real effect.

**Finding 2 — THE SHUFFLE CONTROL BEATS THE CANDIDATE, and that is the
headline of this section.** synth3x-shuf — the *control*, running the OLD
lr 1e-3 recipe with no dropout and no cosine, on synthetic rows whose word
order was deliberately destroyed — posts 2.5500, the best number in the
entire sweep. Same config, same seed, real word order (synth3xT): 2.706.
Destroying the word order didn't cost 0.16 nats; it *gained* 0.16, against
a 3-seed spread of 0.008 on the real-order side. And it statistically ties
the tuned-everything candidate (2.550 vs 2.581 mean, gap inside both seed
spread and split noise).

What this means, said plainly because ro was told voice replication was
crucial: **the measured benefit of the synthetic corpus has nothing to do
with preserving ro's voice, cadence or sentence structure.** A bag of
corpus words in scrambled order regularises at least as well as carefully
spliced in-voice recombinations. The mechanism the numbers support is
dilution: extra in-vocabulary text delays memorisation of the 326 real
train rows, holding the model longer in the regime where it is still
learning spelling, word shapes and function-word statistics — which is
most of what a 25 KB val split measures. Semi-coherent Markov rows are
themselves memorisable; word salad is not, which is a plausible reading of
why the shuffle does *better*, though with one seed we do not lean on that
gap, only on "no worse". Voice-preservation may still matter for the
*sampled text* (the shuffle model's probe samples in §6 are, if anything,
the most word-like of the sweep), but as a val-loss lever it is
unsupported.

**Finding 3 — more is not better.** 5x real-order synthetic (2.687 @ 200,
then monotonically worse) underperforms 3x at the same recipe. The dilution
benefit saturates and the synthetic distribution starts to dominate the
mix. 3x is the measured sweet spot of the scales tried.

## 8. Recommended defaults

Landed in `babble/config.py`, every one env-flippable, shipped by this PR
rather than silently applied to a running install:

| setting | old | new | evidence |
|---|---|---|---|
| `BABBLE_LEARNING_RATE` | 1e-3 | **3e-4** | −0.14–0.18 nats, 3 seeds (§3) |
| `BABBLE_DROPOUT` | 0.0 | **0.2** | part of the replicated winner combo (§3/§4) |
| `BABBLE_TRAIN_COSINE` | off | **on** | −0.05 alone (§3); schedule the winner lanes ran |
| `BABBLE_TRAIN_SYNTHETIC` | off | **on** (3x) | −0.07, 3 seeds vs 3 seeds (§7.2) |
| `BABBLE_TRAIN_STEPS` | 400 → 1000 (interim) | **1600** | matches winner lanes; cosine length is part of the recipe |
| `BABBLE_TRAIN_MIN_STEPS` | — | 600 | early-stop floor (§1) |
| `BABBLE_TRAIN_STALL_MARGIN` | — | 0.05 | measured noise band (§1) |
| `BABBLE_POST_LEARNING_RATE` | (=pretrain lr) | **1e-4** | §5 grid |
| `BABBLE_POST_REHEARSAL` | — | 0.5 | §5 grid |
| `BABBLE_POST_MIN_PAIRS` | — | 100 | §5 |
| `BABBLE_POST_GATE_MARGIN` | — | 0.05 | §5; the gate that makes regressions unshippable |
| `BABBLE_POST_LAYOUT` | pair | **continuation** | §5 mechanism 2 |

On the elephant in §7.2: the single best number in the sweep belongs to the
shuffle *control* (2.5500), which ran the OLD recipe. We did not crown it.
The call, and the reasoning: 2.550 (one seed) vs 2.581 (mean of three, best
seed 2.5706) is a gap of 0.03 — inside the split-noise IQR (0.074) and
inside the candidate's own seed spread (0.029). One unreplicated run does
not beat three replicated ones on a coin-flip margin, and each lever in the
winner config carries its own independent, multi-seed evidence. What the
control *does* prove is mechanism (dilution, not voice), and that is
recorded in §7.2 in bold. If someone wants to chase the last 0.03, the
experiment is one flag away: shuffle the synthetic rows under the winner
config and run three seeds. We kept the real-order generator as the default
because it measured equal-or-slightly-worse only under the *old* recipe,
was never measured worse under the shipped one, and its rows are the ones
a human can read, audit for consent/blocklist purposes, and recognise as
in-voice.

## 9. Honesty section

- **A trigram still beats everything here.** The interpolated byte trigram
  scores 2.23 on the same held-out rows; the best transformer config in
  this entire report is 2.55–2.58. Months of transformer knobs have not yet
  matched a counting model that fits in milliseconds. The transformer keeps
  its job because it *generates* usable novel text and is the vehicle for
  the pair/post-train mechanism, not because it wins the likelihood race at
  this corpus size.
- **The synthetic gain is regularisation, not voice.** §7.2's shuffle
  control says so directly. "Mimics yet expands" turned out to work through
  "expands"; the mimicry is unmeasurable in val loss at this scale. ro
  should read Finding 2 before crediting the Markov generator's
  craftsmanship.
- **Levers that did ~nothing** (all inside the 0.074 noise band, §4):
  capacity 150k–3.3M in either direction, weight decay 0–0.1, any single
  regulariser alone. The 20× parameter range costing nothing is itself the
  result: capacity was never the constraint.
- **What would actually move the needle: more corpus.** Every curve in this
  report turns on the same axis — how long training can run before the 326
  real rows are memorised. Synthetic dilution buys ~0.07; the gap to the
  trigram is ~0.35. Rows in the bank are the only lever with headroom, which
  is exactly why the bot keeps collecting and why post-train is now gated
  rather than trusted.
- Val numbers throughout are nats/byte on an 81–83-row held-out split with
  measured IQR 0.074 across resamplings; single-seed deltas under ~0.08
  in any table above should be read as ties.

## 10. Promotion record (2026-08-21 ~04:00 UTC)

ro asked for the new model live, so the no-touch rule on
`~/babble-live/checkpoints` was lifted for exactly this step:

- **Backup first**: `~/babble-live/checkpoints.bak-promotion-2026-08-20` is a
  full copy of the pre-promotion dir (old pretrained/latest, states, ckpts).
- **Promoted atomically** (temp-file + rename, `latest.pt` last so the bot's
  mtime hot-reload never sees a mismatched set): `pretrained.pt`,
  `train_state.json`, `post_state.json`, `latest.pt` from the final scratch
  run, plus the train-only `data/synthetic_corpus.jsonl` (1200 rows).
- **Consistency checks**: promoted `latest.pt` sha256 matches
  `post_state.json.latest_hash` (`8519c572…`); `last_trained_rows=415` equals
  the live corpus row count and `last_trained_pairs=47` the live pair count,
  so the auto-triggers correctly see nothing new; the state-file key sets are
  identical to what the old live code reads.
- **Verified after**: the live install's own venv and *old main-branch code*
  load the promoted checkpoint and produce the §6.1 samples verbatim;
  `babble-bot.service` stayed active throughout; the log shows normal intake
  (`bot.dropped reason=not_addressed/author_is_bot` — the consent/addressing
  gate doing its job) and `update.noop` ticks on main. Collection is the
  same code path it was yesterday — nothing in the promotion touched intake,
  consent, or correction recording.

**The one live risk left, stated plainly:** until this PR merges, the live
box still runs the *old* trainer. Its auto-pretrain refires at +100 corpus
rows and its **ungated** post-train refires at +10 correction pairs — and
running old post-train on the new checkpoint would ship exactly the kind of
regression this report just fixed. Ten corrections is one enthusiastic
evening. **Merge before playing with corrections**, or the gate isn't
protecting anything yet.

## 11. What ships in this branch, and what deliberately does not

Review of this PR caught the branch carrying ~1.2 GB of experiment
checkpoints and four recombined-corpus files under `experiments/`. Both are
corpus-derived, and the call is that neither may be published:

- **`.pt` checkpoints** — a 3.3M-param model on a ~24k-char corpus memorizes
  rows outright (§5 measured exactly that), so publishing weights is
  publishing corpus text with extra steps. All 52 experiment checkpoints are
  purged from the branch (history included) and `*.pt` is now ignored
  repo-wide. The loss curves (`experiments/**/*.jsonl` metrics) and `.log`
  files — numbers only — stay, and they are what every table in this report
  is built from.
- **`experiments/synthetic_corpus_*.jsonl`** — every row is a verbatim
  splice of real corpus phrasing, the same class as `data/corpus.jsonl`,
  which the repo has never published. Purged and ignored likewise;
  `experiments/gen_synth_corpora.py` regenerates them deterministically from
  the local `data/` when the experiments need re-running.

Probe completions quoted in §6 remain: model output is what the bot posts
publicly by design, and that is the same exposure it gets in the report.
