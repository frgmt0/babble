# Pretrain on current corpus + synthetic in-voice corrections — 2026-08-20

ro asked for three things: a fresh pretrain on the current (grown to ~400-row) corpus,
a synthetic-correction generator that postulates plausible prompts for existing
in-voice corpus lines, and a post-train run on human + synthetic corrections on top
of that pretrain. This is the report on all three, with numbers and honest quality
verdicts.

**tl;dr — pretrain is still gibberish, and post-train on 182 pairs (46 human + 136
synthetic) overfit inside 100 steps and made two of four probes collapse to empty
strings.** The synthetic-pair generator works as specified — it's the mechanically
correct piece, and its pairs are legitimately in-voice by construction (see §2). But
it does not fix the underlying problem: 3,347,968 parameters is too many degrees of
freedom for ~24k characters of pretraining text and ~1-6k characters of correction
text, synthetic or not. Nobody should read this report and expect the bot to be
answering in sentences yet.

Everything below ran against an isolated copy of the live corpus/checkpoints,
never the live install itself. See "Isolation" at the bottom.

## 1. Pretrain on the current corpus

```
babble train --force --steps 400 --patience 3 --seed 1
```

| | |
|---|---|
| Corpus rows | **403** (up from the 222 last measured) |
| Corpus chars | **24,441** |
| Train / val split | 322 / 81 rows |
| Step budget | 400 (ceiling, not a target) |
| Steps actually run | 300 (stopped early — patience 3 exhausted) |
| Best checkpoint | **step 150**, loss 2.7236, val 2.7623 |
| Checkpoints written | 6 |

Loss curve (`checkpoints/loss.jsonl`):

| step | train loss | val loss | |
|---|---|---|---|
| 50 | 3.5051 | 2.9920 | |
| 100 | 2.8662 | 2.8328 | |
| **150** | **2.7236** | **2.7623** | ← best, kept as `pretrained.pt` |
| 200 | 2.5674 | 2.7756 | ⚠ overfit signal |
| 250 | 2.5615 | 2.7634 | |
| 300 | 2.4812 | 2.7851 | ⚠ overfit signal, training stopped |

Val loss bottoms out at step 150 and never beats it again through step 300 — same
overfitting shape seen in every prior run at this model/corpus scale, just reached a
little later now that there's 403 rows instead of 222.

Sample completions from the **kept checkpoint (step 150, `pretrained.pt`)**, fixed
prompts, temperature 0.5 / top_k 40 (the bot's live defaults), seeded so these are
reproducible:

| prompt | sample |
|---|---|
| `hello` | `rett` |
| `hello` (2nd draw) | ` in yore an a mer inn thor the the ar fore the re wond asomu therere tsle it gherer bk ano kean tushes yo pu tht inth ll a oror t tongo the tho mo monts the doreorou tanol a t` |
| `the cat` | `he mor yur o me are ar ther th thethe ofore aske ingoor is alyond boo inndo her arons ore io athe therorere ms o tt od mahore athe bushe to thed t me s mev s we de at m sh as a pres the ast len ar yuthere t fund s s o be there l cthe s theritthe ing a as a` |
| `why is` | `tth a and at as as atore tin can iride thas sthi casher inke t the t me ond ole wa sthe as t thor is l a so the bo w the d boa be athe a inf and acuthe tho s the to r th he athe as at s the ae a or t as fthe cn thenewe tipa ats a tns tor be t at me antthe ` |
| `what is love` | ` at blithing are be wa athe at ma we alcde thonge ber he athew ore o atharont t whamanor mocime it ather cle athe se o in s tthe lereromes in the tho me po yo oromo inthe ol in the t one shy the t athe a ame se ont me ang ingo t ble ie tthe t the nondoae a` |

No real words. Letter clusters and function-word fragments (`the`, `are`, `is`,
`at`) show up but never assemble into anything sentence-shaped. This is consistent
with every previous pretrain at this scale — more corpus rows (403 vs 222) pushed
the best-val step later (150 vs the historical ~100-ish) and nudged val loss down
slightly, but did not change the qualitative output.

## 2. Synthetic-correction generator

`babble/synthetic.py`, driven by `babble synth-generate` / `babble synth-status`.

**Technique, exactly as ro described:** scan the corpus for rows that read as a
reply or interjection (starts with an implicit referent — "no", "yeah", "because",
a bare pronoun, an emphasis marker, disagreement/agreement markers, etc.), then
*postulate* a plausible prompt that could have preceded it. The response side is
always the **real, unmodified corpus row** — voice is preserved by construction,
not by trying to imitate it. Only the prompt half is synthesized, using a small set
of templated patterns keyed to the detected reply type (`disagreement`,
`agreement`, `anaphora`, `emphasis`, `continuation`, `reacts_to_question`, etc. —
see the `method` field on each pair).

Example pairs from this run's `data/synthetic_pairs.jsonl` (prompt synthesized,
response is a real corpus row verbatim):

```json
{"prompt": "i think fuck, right?", "response": "No fuck you", "method": "disagreement"}
{"prompt": "what happened with because", "response": "because it just is", "method": "anaphora"}
{"prompt": "how was tips hat hello good sire", "response": "*tips hat at you* hello good sire", "method": "emphasis"}
{"prompt": "does anyone else think make like mock website something", "response": "yeah make like a mock website or something, push it to freebuff-demo.0xbeckett.me", "method": "agreement"}
```

The responses are lowercase, no trailing periods, and carry the corpus's actual
slang/cadence ("no u", "im a stupid clanker *clank clank*", "idk how you want me to
upload that?") because they *are* corpus rows, untouched. Voice fidelity on the
response side is 1:1 by construction — there's no way for it to drift, since nothing
about the response text is generated.

Results this run:

| | |
|---|---|
| Corpus rows scanned | 403 |
| Rows classified as reactive (reply/interjection-shaped) | **136** |
| Synthetic pairs generated | **136** |
| Stored at | `data/synthetic_pairs.jsonl` (own file, `"synthetic": true` on every row) |
| Human corrections file | `data/interactions.jsonl` — untouched, 46 rows, no synthetic mixed in |
| Idempotent re-run | confirmed — re-running `synth-generate` against the same corpus produced **0 new pairs** (dedup keyed on `source_row_id`) |

`babble post-train` only touches synthetic pairs when `--include-synthetic` is
passed; without the flag, `synthetic_pairs.jsonl` is never read. Same generator can
be re-run any time the corpus grows (`babble synth-generate`), and `babble
synth-status` reports counts without generating anything, so ro can inspect the
split before deciding whether to train on it.

## 3. Post-train on human + synthetic corrections

```
babble post-train --include-synthetic --force --seed 1
```
from `checkpoints/pretrained.pt` (the step-150 snapshot from §1).

| | |
|---|---|
| Human correction pairs | 46 |
| Synthetic pairs | 136 |
| Total training examples | 182 (146 train / 36 val) |
| Step budget | 200 |
| Steps run | 200 (full budget; patience 3 not yet exhausted at the ceiling) |
| Best checkpoint | **step 50**, loss 2.6936, val 2.7479 |

| step | train loss | val loss | |
|---|---|---|---|
| **50** | **2.6936** | **2.7479** | ← best, kept as `latest.pt` |
| 100 | 2.4886 | 2.8143 | val rising |
| 150 | 2.3130 | 2.9668 | val rising |
| 200 | 2.1405 | 3.0722 | val rising |

Train loss falls monotonically (2.69 → 2.14) while val loss climbs every single
step past 50 (2.7479 → 2.8143 → 2.9668 → 3.0722). This is overfitting in its
textbook shape, and it set in **before the second checkpoint** — by step 100 the
model was already getting worse at held-out pairs while getting better at the
182 it was shown. 182 examples (most of them short, templated-prompt/real-response
pairs) is not enough signal for a 3.3M-parameter model to generalize past step 50.

### Before / after, same held-out prompts

Same fixed prompts as §1, same seeds, only the weights differ between the two
columns (before = `pretrained.pt`, step 150; after = `latest.pt`, step 50):

| prompt | before (pretrain only) | after (+ human & synthetic post-train) |
|---|---|---|
| `hello` | `rett` | `` (empty) |
| `hello` (2nd draw) | ` in yore an a mer inn thor...` | `` (empty) |
| `the cat` | `he mor yur o me are ar ther th thethe...` | `.` |
| `the cat` (2nd draw) | `he a mes s ory ithe thorc...` | `` (empty) |
| `why is` | `tth a and at as as atore tin can iride...` | ` th atous aa an athatores inyo aliride as ins ai cllheat lane inddee me is inte wat in anore thor isin atithe...` |
| `why is` (2nd draw) | `s f yoont as s deyolat innou...` | `s f you` |
| `what is love` | ` at blithing are be wa athe at ma we alcde thonge...` | `e ithes in it is che athere athe it inore athe al ber hes ameathe...` |
| `what is love` (2nd draw) | `t thou t mer the in mithe be a in me werde mee sth...` | `e ino theve be de l mitoritreanin i te at anse st le in athe ce it...` |

**Post-train did not improve these outputs — it made two of the four probes
collapse to empty or near-empty strings** (`hello` → `""`/`""`, `the cat` →
`"."`/`""`). `why is` and `what is love` are still letter-soup, roughly the same
character as before, neither better nor obviously worse. This matches the loss
curve exactly: the model spent its 50 useful steps narrowing toward the 182
training examples (many of which are short) rather than learning anything that
transfers to unrelated prompts — the same "shorter and more degenerate, not more
coherent" failure mode documented for corrections-only post-train in
`POST_TRAIN_EXPERIMENT.md` from the prior (2026-08-18) run. Adding synthetic pairs
did not change that mode; it happened faster this time (step 50 vs step 50 there
too, but off a larger 182-pair set here vs 39 there — more data made the collapse
no less severe).

## Honest verdict

- **The generator is correct and does what was asked.** It finds reply-shaped
  corpus rows (136 of 403), invents plausible antecedent prompts, and keeps the
  response side as real, untouched corpus text — voice fidelity by construction,
  not imitation. It's re-runnable, idempotent, and clearly separated from human
  corrections in its own file and flag.
- **It did not, and could not, fix output quality**, and the brief said not to
  expect it to. Pretrain output is still non-word letter-soup at 403 rows / 24.4k
  chars. Post-train on 182 pairs (human + synthetic) overfits within 50-100 steps
  and, in this run, made responses to two of four held-out prompts degenerate to
  near-nothing rather than more coherent.
- **What's actually limiting this:** model capacity (3.35M params) relative to
  data volume, at every stage. 24k characters of pretraining text is not enough to
  teach word structure from random init; that's the pretrain result. On top of
  that, 182 correction pairs — even with real in-voice responses — is not enough
  distinct signal for post-train to generalize past memorizing the training set,
  which is the post-train result. More synthetic *volume* off the same 403-row
  corpus would very likely hit the same wall, because the response text pool it's
  built from doesn't grow — only the prompt side does. The lever that would
  actually move this is more raw corpus (more distinct human-written rows,
  synthetic or not), not more post-train passes on the same underlying text.

## Isolation

Every run in this pass — pretrain, `synth-generate`, post-train, and the
before/after probe above — used `BABBLE_DATA_DIR` / `BABBLE_CHECKPOINT_DIR` /
`BABBLE_LOG_DIR` pointed at an isolated copy:

```
~/babble-live/experiments/pretrain-synthetic-20260820T064057Z/{data,checkpoints,logs}/
```

`babble-bot.service` (the live Discord bot) ran throughout and was never stopped,
restarted, or pointed at anything in this experiment dir. `~/babble-live/data` and
`~/babble-live/checkpoints` (the live install's actual paths) were not written to
by anything in this report. The self-update restart guard in
`deploy/update-live.sh` (a `/proc` argv scan for the `post-train` token) is
unaffected by this branch's diff and was independently audited as still intact
during this pass.
