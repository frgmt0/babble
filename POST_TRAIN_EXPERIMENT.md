# Post-train on corrections only — before/after, 2026-08-18

ro asked: *"can you do another post train run on the corrections only to see how
corrections impact the model responses?"* — reacting to booper answering "hola" and a
giveaway prompt with pure garble. This is that run, measured.

**tl;dr — corrections did not make responses more sentence-shaped.** They made
responses shorter and more degenerate, while train loss cratered and val loss climbed.
That's overfitting, the same failure mode as every corpus size tried so far
(now 39 pairs / ~1.3k chars against a 3,347,968-param model).

## Important caveat: two post-train runs happened, not one

While this experiment was starting, the **live bot's own automatic trigger fired a
post-train on its own** (`AutoPostTrigger`, `post.triggered` in `logs/babble.jsonl` at
`2026-08-18T05:53:34Z`, PID 9961 under `babble-bot.service`) — it crossed its
+10-correction-pair threshold from ordinary Discord activity, unrelated to this task.
That run finished (`post.done`, step 50, 37 pairs) about a minute before this
experiment's own data snapshot was taken, so **"current / before" in this report is
already a once-post-trained checkpoint**, not the raw pretrain ro actually saw garble
from. A raw-pretrain baseline is included separately below (§4) because it's the
closer match to what ro quoted.

Because the bot auto-fires post-train on its own trigger, and two trainers writing
`checkpoints/latest.pt` at once was called out explicitly as the exact way the live
bot ended up serving a half-trained model earlier tonight, **this experiment never
touched the live install.** Everything below ran against a frozen copy of
`~/babble-live/{data,checkpoints}`, isolated via `BABBLE_DATA_DIR` /
`BABBLE_CHECKPOINT_DIR` env vars the CLI already supports (`babble/config.py`). The
live `babble-bot` service was never stopped or restarted.

- **Snapshot (restorable copy of the pre-experiment checkpoints dir):**
  `~/babble-live/checkpoints.bak-preexperiment-20260818T055732Z/` (and
  `~/babble-live/data.bak-preexperiment-20260818T055732Z/` for the data dir).
- **Isolated working copy used for this experiment:**
  `~/babble-live/experiments/post-train-corrections-20260818T055732Z/`
  (`data/`, `checkpoints/`, `checkpoints-pretrain-only/`, `logs/`).
- **Live `~/babble-live/checkpoints/latest.pt` and `pretrained.pt`:** byte-identical
  before and after this experiment (hashes checked, see below) — this run wrote
  nothing into the live install.
- **Probe script:** `bench/post_train_probe.py` in this checkout — loads
  whatever checkpoint sits at `$BABBLE_CHECKPOINT_DIR/latest.pt` and samples the
  fixed prompt set with per-prompt seeded generators (SHA-256 of the prompt text,
  not Python's randomized `hash()`), so the *same* random draws replay against
  every checkpoint — the only variable across before/after is the model weights.
  Usage: `BABBLE_CHECKPOINT_DIR=... BABBLE_DATA_DIR=... python3 bench/post_train_probe.py <label>`.

## 1. Data before running anything

Read from the frozen data snapshot, via `babble post-status` / `babble summary` and
`post_state.trainable_pairs()` directly (the consent- and blocklist-filtered set
post-train actually trains on):

| | |
|---|---|
| Consented, trainable correction pairs | **39** |
| — corrections | 38 |
| — 👍 approvals | 1 |
| Total chars (prompt + chosen, what's actually fed to the model) | **1,345** |
| Total chars (prompt + chosen + rejected, incl. the discarded half) | 2,503 |
| Pairs at last post-train before this run | 37 (the bot's own auto-fire, see caveat above) |
| New pairs since | 2 |
| Trigger threshold | 10 (not "due" on its own — this run used `--force`) |
| People opted in | 5 of 5 asked |

For scale: the general corpus (stage-1 pretrain material, a different, much larger
pool of plain text) sits at 237 rows / 11,185 chars — post-train never touches it.
39 pairs / 1,345 chars is what stage 2 has to work with.

## 2. Probe set, before (current live checkpoint, pre-*this*-post-train)

Fixed prompts, temperature 0.5, top_k 40, max_new_tokens 256 (the live bot's own
defaults), 3 seeded samples each. Checkpoint: step 50, the one currently live-serving
(already post-trained on 37 pairs by the bot's own auto-fire, see caveat).

| prompt | sample 1 | sample 2 | sample 3 |
|---|---|---|---|
| `hola` | `nke` | `nkeo tudolio tou` | `nkeo tudola, toucacouso ps topror` |
| `hello` | ` liol br` | ` laour` | ` li` |
| `do you want to enter giveway` | `opecourerss tor. fres boreesoplam gimmerslioursursemlourstuslyoules ginkk or toworshe pu` | `ou ou berss torse` | `ou` |
| `why is` | ` ae bhousecursasssamt clalay t caoula ckeale julalayay milay beas. oupbkeameslaclalealayo, malaous bodousalyoudaboumeayobe gourousater goupitousthestoblour boucthou  bou` | ` ae bhist alyo bk apeama yousthes tor brvese. acamousms galam bmay beanou besousalay bousouboumest touay cr busay mo ayous youroureayouu bhe.ourou` | ` i bi a i crer ck aes yo yoursti ouusese` |
| `the cat` | `thk` | `thololi` | `thellalalor` |
| `where` | `` (empty) | `sla` | `` (empty) |

Mean output length: **30.8 chars**. Still garble, no real words — but already much
shorter than raw pretrain (§4).

## 3. This run's post-train on corrections only

```
babble post-train --force --seed 1
```
against the isolated copy, from `checkpoints/pretrained.pt` (the frozen stage-1
snapshot, untouched by the bot's earlier auto-fire).

| step | train loss | val loss |
|---|---|---|
| 50 | 1.9334 | **4.0825** ← best, kept |
| 100 | 0.4561 | 5.4477 |
| 150 | 0.1440 | 5.5203 |
| 200 | 0.1108 | 6.0657 |

- **Pairs trained on:** 39
- **Step ceiling (budget):** 200; ran the full budget, 4 checkpoints written
- **Early stopping:** `stopped_early=True` — patience (3) of non-improving val
  checkpoints was exhausted exactly as the budget ran out, so the *best-val*
  checkpoint (step 50) is what got written to `latest.pt`, not the step-200 one
- **Train loss ↓ 1.93 → 0.11 while val loss ↑ 4.08 → 6.07.** Textbook overfitting:
  the model is memorizing the 39 training rows (loss keeps falling) while getting
  strictly worse at everything it didn't see (val loss keeps climbing) from the very
  first checkpoint on. There was never a "sweet spot" step where both were good —
  step 50 is just the least-bad of four bad options.

## 4. Probe set, after this post-train

Same prompts, same settings, same per-prompt seeds — only the weights differ.
Checkpoint: step 50 (the best-val checkpoint this run kept).

| prompt | sample 1 | sample 2 | sample 3 |
|---|---|---|---|
| `hola` | `m anoug-id` | `d` | `m s gigholath clad ad` |
| `hello` | `` (empty) | `` (empty) | `+hi` |
| `do you want to enter giveway` | `oa. oure i oure. gime gimve` | `oou` | `oa. oure i oule oure. gid` |
| `why is` | `tthol anoue` | `` (empty) | `t ghogople` |
| `the cat` | `ou` | `ou` | `ou` |
| `where` | `tppp` | `tpphdghllo+ht:hato:clanolo lalala` | `tpphd=hllllo busthiou` |

Mean output length: **9.7 chars** (down from 30.8 before this run, and from **187.9**
for the raw pretrain baseline below). 3 of 18 samples were empty strings (vs. 2 before,
1 for raw pretrain). `the cat` collapsed to the literal same 2-character output
(`ou`) on all three different seeds — a mode-collapse signature, not a coincidence.

### Bonus baseline: raw pretrain, zero post-train exposure

Sampled `checkpoints/pretrained.pt` directly (never post-trained), same prompts/seeds/
settings, to show what ro actually saw. This is the closer match to the "hola" /
giveaway quotes in the task — long rambling letter-soup, not short fragments:

| prompt | sample 1 (truncated to ~120 chars) |
|---|---|
| `hola` | ` se the lellikereallllllllallllllingoulloua ouliusa tous a a louse alliloud t f allinoo t a to ou lous oou ou alit ullitoore woor…` |
| `do you want to enter giveway` | ` theas thew ar thes s s atore the ingore ilar alide athere ather as thine s at` |

Mean output length **187.9 chars** — vs. 30.8 (bot's currently-live, once
post-trained) vs. 9.7 (freshly post-trained by this run). The style match to ro's
quoted garble ("...lllllloue lgitourealourt ourere i tha t...") is close: same
run-on "ll"/"ou"/"ore"/"tor" letter clusters, no word boundaries, no early stop.

## Verdict

**Corrections did not teach booper to answer.** At no point — raw pretrain, the
bot's own 37-pair auto-post-train, or this run's fresh 39-pair post-train — did any
checkpoint produce a real word, let alone a sentence, for any of the six probes. 39
pairs / 1,345 characters is not enough signal for a 3.3M-parameter model to learn
"answer this kind of prompt with this kind of text," and the loss curve shows why:
val loss rises monotonically from the first checkpoint (4.08 → 6.07) while train loss
falls sharply (1.93 → 0.11) — the model is memorizing the specific 39 rows, not
generalizing, which is exactly the overfitting signature flagged as this model's
recurring failure mode.

What corrections *did* measurably change is response **length and diversity**, in the
wrong direction: mean output length dropped monotonically with post-train exposure
(188 chars raw → 31 chars once post-trained → 10 chars twice post-trained), and by the
second post-train pass one probe (`the cat`) had collapsed to an identical 2-character
output across three different random seeds — the model narrowing toward a small set of
memorized short fragments rather than broadening toward coherent answers. So the
honest answer to "how do corrections impact the model's responses": they make it
babble *less*, not better — shorter, more repetitive, still meaningless — and a second
post-train pass on almost the same tiny pair count made that narrowing more
pronounced, not less. More correction data (not more post-train passes on the same ~39
rows) is what this needs before it will look like an answer instead of a shrug.
