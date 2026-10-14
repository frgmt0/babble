# longctx-v2 ("unhinged") data report — 2026-10-14

Data for the next SFT run, **longctx-v2** (`configs/sft/unhinged-mac.json`). It continues from the live model
`hf-booper-longctx-v1` (`runs/longctx-v1/export`; `model-int8.safetensors` sha256 `c543cb9d…1e35`, same as the
promote report). qa-v1 was promoted on 2026-10-02 01:21 and rolled back to longctx-v1 at 08:51 the same day, so
it is not a base. The aim is better conversation (follow the thread, answer what was said, multi-turn) and more of
booper's chaotic Discord energy, with the same personality. Nothing was trained, promoted or pushed for this report.

Every number here comes from a run on 2026-10-14 on the M2 Pro / 16 GB Mac. Discord-Dialogues is a public
Apache-2.0 dataset, so its examples are printed by `sft/vibe_report.py`. For our own users' snapshot
(`data/server-snapshot-2026-10-14/`, gitignored, already consent-filtered and pseudonymised by `babble export`),
only aggregates appear here.

## Tools

- `sft/vibe.py`: `vibe_score` / `vibe_features` (a transparent, hand-weighted per-turn score), `engagement_reason`,
  `conversation_verdict` (the conversation filter), and the snapshot readers (`local_snapshot_records`,
  `corpus_prompts`, `snapshot_sha256`). Pure, no network.
- `sft/vibe_report.py`: streams Discord-Dialogues at the pinned revision `a8b2294…` and reports pass rates, a
  threshold sweep, score distributions, rejection reasons, a supply scan, and the snapshot aggregates.
- `sft/vibe_eval.py`: before/after replies to real user prompts, scored for vibe and coherence.
- `sft/sft_longform.py`: new sources `discord_vibe` and `booper_local` (see `sft/README.md`, "Unhinged run").

## How earlier runs used Discord-Dialogues

The dataset has **7,300,966** rows (parquet metadata). It is ordered by conversation length, longest first. From the
`turns` column: 68,742 rows have 6+ turns, 96,751 have 5, 304,428 have 4, 1,038,432 have 3 and 5,792,613 have 2. Only
the first **469,921** rows (4+ turns) can give more than one exchange. `_collect_groups` reads from the head of the
stream with no shuffle, so every run took a prefix.

These row counts are measured, not estimated. Each comes from replaying the `discord` source's collection
(`_turn_records` with gif handling) until the preset's want (`int(examples*w/total)` + val) is reached:

| run | discord targets wanted | stream rows read | distinct conversations used |
| --- | ---: | ---: | ---: |
| qa-v1 (rolled back) | 70,020 | 32,138 | 19,556 |
| multiturn-v1 | 99,550 | 45,707 | 28,928 |
| longctx-v1 (live) | 294,860 | **233,989** | **116,699** |

Every slice is a prefix of the next, so **116,699** distinct conversations is everything any earlier run used, about
1.6% of the rows. longctx-v1 trained 0.19 of an epoch, so it saw only part of even those. Story-v1/v2 ran older pair-layout code and are not replayed
here, but they also read from the head, with smaller discord wants than longctx-v1.

**longctx-v2** reads 315,877 conversations as `discord_vibe` from rows 233,990 to 6,701,929, all of them new to
every earlier run. Its `discord` rehearsal source takes 103,203 conversations, all inside longctx-v1's prefix. That
is **419,080** distinct conversations in total, 3.6x what longctx-v1 used.

## The filter on real data

`vibe_score` gives a dry, sentence-case reply 0. Lowercase is worth +0.5, and laughter, emoji, elongation, caps
bursts, slang, `!!`, playful markers and gif tags add more. Assistant-speak is −3, URLs and spam −2, copypasta −1.5,
a bare filler reply −1, and walls of text up to −3. `conversation_verdict` rejects a conversation in this order:
blocklist hit, too few targets, more than 50% one-word turns, any target ≤ −2, fewer than 50% of replies engaging the
previous turn, then mean target vibe < `min_vibe`. Rows that open with the `assistant` speaker (about 45% of the
head) start at the first user turn.

### Score distributions (5,000-row samples)

| sample | assistant targets | mean | p50 | p90 | conversation-mean p90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| head (rows 0-4,999) | 21,261 | 0.328 | 0.4 | 0.9 | 0.70 |
| past longctx-v1 (rows 233,989+) | 9,985 | 0.324 | 0.4 | 0.8 | 0.70 |

Feature hit rates on head targets: lowercase 41.3%, slang 9.8%, caps burst 5.1%, elongated 3.7%, laughter 2.1%,
burst 2.1%, filler 1.5%, emoji 1.2%, playful 0.8%, spam 0.15%, assistant-speak 0.05%. Most of Discord-Dialogues is
plain chat. The high-energy register is the top few percent, so it has to be selected.

### Threshold sweep (pass rate, 5,000 rows)

| min mean vibe | 0.0 | 0.4 | 0.5 | 0.6 | **0.75** | 1.0 | 1.25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head, min_targets 2 | 62.4% | 26.4% | 23.0% | 11.5% | **5.0%** | 1.4% | 0.4% |
| rows 233,989+, min_targets 1 | 75.0% | 35.5% | 32.8% | 10.8% | **5.9%** | 2.0% | 0.7% |
| rows 1,000,000+, min_targets 1 | 44.3% | 25.6% | 24.1% | 4.8% | **4.5%** | 2.4% | 0.9% |
| rows 4,000,000+, min_targets 1 | 53.4% | 30.9% | 28.8% | 7.4% | **6.5%** | 3.4% | 1.6% |

On the head, rejections at 0.75 / 2 targets were low_vibe 2,935, low_engagement 1,773, blocklist 16, bad_target 13
and one_word 13, with 250 passing.

**Chosen: `--discord-vibe-min 0.75`, `--discord-vibe-min-targets 1`, `--discord-vibe-skip-rows 233989`.**
- 0.75 sits above the 90th percentile of conversation-mean vibe (0.70). A reply cannot reach it on lowercase alone
  (+0.5), so every kept conversation has real energy markers. At 0.6 the pass rate doubles, but most of the extra is
  lowercase with one slang word. At 1.0 the sample pass rates are 2.4-3.6x lower than at 0.75. Scaling the full scan's 328.7k targets by that ratio gives roughly 90-140k targets (an estimate, not a run).
- min_targets 1: past row 469,921 every row is a single exchange, and requiring 2 would cap the source at the
  ~26k targets in rows 233,990-469,900. Single exchanges still must engage the turn before them.
- skip 233,989: exactly where longctx-v1's `discord` slice ended (measured above).

### Supply (measured)

The `--supply-rows 700000` scan from the head at min_vibe 0.75 / min_targets 2 found 13,063 conversations (29,248
targets) inside longctx-v1's prefix. Past it there are 12,780 (25,560 targets), and no multi-exchange row after row
469,900. With min_targets 1, rows 233,990-700,000 give 26,458 conversations / 39,238 targets.

The full build of the preset (`--prepare-data`, examples 630,000, val 2,000) gave these numbers for `discord_vibe`.
It scanned **6,467,940** rows and kept **315,913** conversations (4.9%). Rejections: low_engagement 3,077,136,
low_vibe 3,050,981, one_word 14,628, blocklist 5,246, bad_target 3,935, too_few_targets 101 (a URL target dropped
after the verdict). Another 28 conversations were skipped because this run's `discord` already held the same content
(`_skip_seen`), and 8 duplicate groups were dropped. That leaves 327,586 train and 1,040 val targets. 12,700 train
targets carry history (the multi-exchange rows), and gif tags: 562 train targets (573 synthetic, 2 real).

## Our users' snapshot (aggregate only)

`corpus.jsonl` (live 603 rows, boopit 33). The blocklist skipped 0 rows.

| source | n | mean vibe | p50 | p90 |
| --- | ---: | ---: | ---: | ---: |
| mention | 540 | 0.300 | 0.5 | 0.5 |
| dm | 35 | 0.009 | 0.5 | 0.5 |
| reply | 6 | 0.617 | 0.5 | 1.7 |
| ambient | 48 | 0.354 | 0.5 | 0.9 |
| correction | 7 | 0.614 | 0.5 | 1.3 |
| **direct prompts (mention/dm/reply)** | **581** | **0.286** | **0.5** | **0.5** |

Direct-prompt feature rates: lowercase 61.3%, caps burst 3.6%, slang 2.1%, filler 1.7%, elongated 1.6%, emoji
1.6%, URL 1.4%, playful 1.0%, spam 1.0%, laughter 0.7%, copypasta 0.5%, wall of text 0.5%, burst 0.3%. Histogram:
[-1,0) 12 (plus 14 below -1), [0,0.5) 187, [0.5,1) 349, [1,1.5) 16, [1.5,2) 3.

The heuristic agrees with how booper's users actually write. They are casual (61% lowercase, more than the dataset's
41%) but no more chaotic than an average Discord-Dialogues turn (mean 0.29 vs 0.33). The ≥0.75 slice is therefore
deliberately louder than the people talking to booper, which is the requested "push it further", and the coherence
rehearsal and gate keep it from tipping over. 569 distinct direct prompts are available to `vibe_eval.py`.

`train.jsonl` pairs: live 9 rows. 7 kept (5 corrections, 2 approvals), 2 rejected as meta-instructions (an
"if someone says X, say Y" rule and an explanation of how to answer a question). boopit has 0 rows (`no usable rows`
if passed alone; fine alongside live). One kept correction is flirtatious in tone. It passes the blocklist and was written by the person who prompted it, but promoters may want to look at it; removing
it is a consent/content decision, not a filter change.

## Mix (measured build)

| source | weight | train | val | mean tokens | train tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| discord_vibe | 0.52 | 327,586 | 1,040 | 48 | 15.7M |
| discord | 0.425 | 266,645 | 850 | 86 | 22.9M |
| persona (x3) | 0.015 | 1,734 | 31 | 39 | 0.07M |
| ultrachat | 0.01 | 6,296 | 26 | 1,164 | 7.3M |
| booper_local (x3) | 0.01 | 21 | 0 (train-only) | 27 | 0.0006M |
| oasst | 0.009 | 5,663 | 18 | 588 | 3.3M |
| no_robots (x2) | 0.006 | 3,774 | 12 | 515 | 1.9M |
| smoltalk (multi-turn) | 0.005 | 1,398 | 12 | 1,245 | 1.7M |
| **total** | | **613,117** | 5,359 views | | **52.8M** (20.5M target) |

Rehearsal (ultrachat/oasst/no_robots/smoltalk) is 2.8% of examples, 27% of tokens and about 29% of loss-bearing target
tokens. An earlier draft gave ultrachat/smoltalk 2-3% of examples, and that already made rehearsal 51% of target
tokens, because their answers average 300-600 target tokens against about 20 for a Discord reply. persona and
booper_local are supply-capped (155 hand-written pairs; 7 real pairs), so their nominal weights are upper bounds.
smoltalk lost 460 groups to the cross-source dedupe, as in earlier runs.

Gate: aggregate val must beat longctx-v1, no source may regress by more than 0.05 nats,
`guard_sources discord persona ultrachat`, and `multiturn_must_improve discord_vibe`. Baseline (longctx-v1, step 0)
val losses on this split: aggregate 2.1359, discord 2.539 / multi-turn 2.506, discord_vibe 2.553 / multi-turn 2.611,
persona 2.456, ultrachat 1.409, oasst 1.468, smoltalk 1.239, no_robots 1.817.

## Timing (deadline Fri 2026-10-16 04:00 PDT, hard stop 07:00)

Measured on this Mac with the real preset, the real base and the full cached data (duty 1.0, `taskpolicy -c utility`):

| item | measured |
| --- | --- |
| full data build (`--prepare-data`) | 871 s (14.5 min). The 6.47M-row discord_vibe scan fits inside the 793 s between model load and the vibe log line (the earlier sources stream in that window too), so ≥ 8,100 rows/s |
| standalone filter scan (`vibe_report --supply-rows`) | 4,685-5,216 rows/s |
| baseline eval, 5,359 views | 166 s (+11 s samples) |
| final eval | 177 s |
| training step (16 x 2048-token micro-batches) | 13.4-13.6 s, ~17.3k real tokens/step (~1,280 tok/s) |
| checkpoint save (with `--mps-empty-cache-gb`) | 8-10 s (12 min 23 s without it, under swap) |
| `grad_checkpoint_min_width 1536` (added after the first launch OOMed at step ~55) | +1.2% compute: 254.1 s vs 251.2 s over the first 400 micro-batches (recompute on the 5.9% of micro-batches ≥ 1536 wide) |
| INT8 export + round-trip | ~4 s |

The longctx-v1 report's 2,250 tok/s at duty 1.0 is not comparable. This data has far shorter examples, so the
fixed-shape micro-batches carry fewer real tokens. Per-step wall time is the number that matters.

Assumed: the run starts by **Wed 18:00 PDT** (34.0 h = 122,400 s before 04:00 Fri), and nothing else heavy runs on
the Mac.

- Fixed costs: build 871 s (0 if the cached `runs/longctx-v2/data-cache.pkl` is reused, which it will be with the
  README commands) + baseline eval 177 s + final eval/save/export 200 s = 1,248 s.
- Per step: 13.5 s train + eval every 500 steps (177 + 11 + ~5 s best save = 193 s / 500 = 0.39 s) + checkpoint every
  100 (10 s / 100 = 0.10 s) = **14.0 s**.
- (122,400 - 1,248) / 14.0 = 8,654 steps, so **8,600 steps** = `tokens 281804800` (8,600 x 2048 x 16). The planned
  end is about 33.8 h after the start (≈ Fri 03:47 if started at 18:00), or about 15 min earlier with the cache.
  That leaves roughly 3.2 h before the 07:00 hard stop.
- After the step-55 OOM, `grad_checkpoint_min_width 1536` adds ~1.2% to the train time: 13.66 s train, **14.15 s**
  per step all-in. With the cache reused (fixed costs 377 s), 8,600 steps take 8,600 x 14.15 + 377 ≈ 122,070 s
  ≈ 33.9 h. A relaunch at Wed 17:30 PDT ends ≈ Fri 03:25, still before the 04:00 target and 3.6 h before the hard
  stop. The step count is unchanged.
- 8,600 steps is about 2.6 passes over the 52.8M-token train set (3,268 steps per pass). discord_vibe cannot grow,
  because the whole dataset was scanned. More passes over the same data is the only way to use the full window, and
  the gate exports the best eval, not the last step. Each hour the start slips past 18:00 costs about 257 steps. The
  simplest correction is to start anyway and, if needed, stop and export at 04:00 (below).

If the run has to stop early: `sft/stop.sh`, then
`.venv/bin/python sft/sft_longform.py --name longctx-v2 --base runs/longctx-v1/export --config configs/sft/unhinged-mac.json --local-snapshot data/server-snapshot-2026-10-14/live data/server-snapshot-2026-10-14/boopit --export`.
That packs `runs/longctx-v2/best/`, the last eval (every 500 steps, ≈ 1 h 57 min apart) that beat the base and passed
every gate. It refuses if none has. `--resume` with the same flags continues from the last 100-step checkpoint.

## Before/after eval baseline (`sft/vibe_eval.py`)

longctx-v1 on 200 seeded real direct prompts (seed 0, temperature 0.8, max 120 new tokens, CPU, 34 s):

| vibe mean | vibe p50 | vibe p90 | vibe ≥ 0.75 | coherent | engaged | echo | degenerate | assistant-speak | mean reply tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.183 | 0 | 0.5 | 6.0% | 66.0% | 66.5% | 0.5% | 0.5% | 0% | 14.4 |

"Coherent" = non-empty, not an echo of the prompt, not a loop/spam, and engages the prompt (shared stemmed content
words, answers a question, reacts, acknowledges, or addresses the person). It is a cheap proxy, not a quality
judgement. Per-reply rows (which contain users' messages) are in `runs/vibe-eval/longctx-v1.json`, which is
gitignored. Rerun the same command on `runs/longctx-v2/export` and compare with `--compare`.

## Reproduce

```bash
SNAP="data/server-snapshot-2026-10-14/live data/server-snapshot-2026-10-14/boopit"
.venv/bin/python sft/vibe_report.py --n 5000 --min-vibe 0.75 --min-targets 2 --supply-rows 700000
.venv/bin/python sft/vibe_report.py --n 5000 --skip-rows 233989 --min-vibe 0.75 --min-targets 1 --supply-rows 700000 --snapshot $SNAP
.venv/bin/python sft/vibe_report.py --n 5000 --skip-rows 1000000 --min-vibe 0.75 --min-targets 1
.venv/bin/python sft/vibe_report.py --n 5000 --skip-rows 4000000 --min-vibe 0.75 --min-targets 1
sft/train.sh longctx-v2 --base runs/longctx-v1/export --config configs/sft/unhinged-mac.json --local-snapshot $SNAP --prepare-data
.venv/bin/python sft/vibe_eval.py --model runs/longctx-v1/export --snapshot $SNAP --n 200 --seed 0 --out runs/vibe-eval/longctx-v1.json
```
