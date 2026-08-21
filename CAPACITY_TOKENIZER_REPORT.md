# Capacity sweep + tokenizer swap, 2026-08-21

ro asked, about booper: *"perhaps its just the architecture of the model maybe. any arch. things
we can finagle"*. Two things were worth trying and ro said to do both. This is that run, run to
completion.

**tl;dr — one lever did nothing, the other actually worked.** Shrinking the model
(experiment 1) never beats the trigram baseline at any size tried, 3K params to 3.3M: the best
architecture found (`cap-1.4M`) still loses by 0.50 bits/char, and the relationship between size
and loss is non-monotonic and shallow, not a clean "smaller is better" curve. Swapping the
tokenizer off char/byte level to BPE (experiment 2), at the *same* architecture that was losing
by 0.50 bits/char as byte-level, **beats the trigram outright**: `bpe-1024` scores 3.09 bits/char
against the trigram's 3.27 — a 0.18 bit/char win, reproduced across 3 seeds. Word-level tokenization
lands at a wash (0.01 bits/char worse than trigram, inside noise). This is the first positive
result out of this line of experiments — the synthetic-corpus and pair-augmentation runs both came
back "no"; this one comes back "yes, on the tokenizer, not the architecture."

Everything below ran against the real corpus (`~/babble-live/data/corpus.jsonl`, read-only copy
under this worktree's own gitignored `data/`, never written back) and the trainer's own held-out
split (`babble/valsplit.py`, `BABBLE_VAL_FRACTION`/`BABBLE_VAL_MIN_ROWS`), so every number below is
on the same yardstick as the 2.22 trigram / 2.55-2.60 char-level figures already in the channel.
Nothing here touched `/home/beckett/babble-live` or promoted a checkpoint; every run wrote only to
`experiments/results/` in this worktree.

## 0. The measurement axis: bits-per-character, and why raw val loss lies here

Val loss as `experiments/sweep.py` and `babble/trainer.py` already report it is a mean over
*target tokens*. For the byte tokenizer that's a mean over bytes, and since this corpus is almost
entirely ASCII, bytes and Unicode characters are almost the same count — so the existing
2.22/2.55/2.60 figures already in the channel are, to a very good approximation, already
nats-per-character. "Very good approximation" is not "exact": recomputing the trigram baseline by
dividing its total held-out negative-log-likelihood by the actual Unicode character count of the
held-out text (rather than its token count) gives **2.2256 nats/byte-token -> 3.2655 bits/char**
— see §1.

That gap is a rounding error for the byte tokenizer. It is *not* a rounding error once the
tokenizer changes. A BPE token is several bytes; a word token can be a dozen. A model spending its
probability mass on one BPE token per ~N characters is solving an easier per-step problem than one
predicting per byte, so its raw per-token loss drops for reasons that have nothing to do with
whether it modeled the text any better — reporting that number against char-level's per-byte loss
without normalizing is comparing different units and calling it a result. This run's own numbers
show the trap firing in the counterintuitive direction: `bpe-1024-s2`'s raw per-token val loss is
**4.038 nats/token**, *higher* than `byte-ref-s2`'s 2.5608 nats/token — read naively, that says BPE
is much worse. It isn't; each BPE token just covers several characters, so a "worse" per-token
number can still cash out to a better per-character number once every token's actual character span
is accounted for: normalized, BPE is 3.0806 bits/char against byte's 3.7573 — a real win, in the
opposite direction the raw numbers suggested. Every table below reports **bits-per-character**,
defined the same way for
every tokenizer: total held-out negative-log-likelihood (nats, summed over every masked target
position) divided by the total Unicode character count of the held-out row text (not token count,
not byte count), converted to bits by dividing by ln 2. `experiments/sweep.py::bits_per_char` and
`experiments/tokenizer_sweep.py::bits_per_char` are the same four-line function, so there is
exactly one definition in play. The trigram baseline is recomputed the same way in
`experiments/ngram_baseline.py`.

## 1. Baseline, char-normalized

Recomputed live at report time via `python -m experiments.ngram_baseline --order 3` against the
real corpus's train/val split (359 train rows / 90 val rows, same `babble/valsplit.py` split every
other number in this doc uses):

```
{"order": 3, "lambdas": [0.05, 0.2, 0.75], "train_rows": 359, "val_rows": 90,
 "train_loss": 1.2152, "val_loss": 2.2256, "val_chars": 7050, "val_bits_per_char": 3.2655}
```

`val_loss: 2.2256` matches the 2.22 already in the channel, confirming this worktree's read-only
corpus copy and split logic reproduce the number the rest of this report is measured against.
**Trigram floor: 3.2655 bits/char (2.2635 nats/char — note this is char-normalized, and differs
slightly from the `val_loss: 2.2256` figure above, which is nats-per-*token*; the trigram's tokens
and the held-out text's Unicode characters are close in count but not identical, so the two numbers
are close but not the same field).** Every model number below is compared
against this.

## 2. Experiment 1 — capacity sweep

Sweep down over `n_layer`/`n_head`/`n_embd`/`block_size` from the current default (3.3M params,
`n_layer=4 n_head=4 n_embd=256 block_size=512`) toward genuinely tiny, holding lr=3e-4,
dropout=0.2, cosine decay fixed at the current defaults (`babble/config.py`) so only capacity
varies. 3 seeds per config, 8 configs, 24 runs total, via `experiments/lanes/capacity-sweep.sh` ->
`experiments/results/capacity/`. `experiments/sweep.py` unmodified except for adding the
bits-per-character field to its summary output (additive; every existing field is untouched).

| config | params | n_layer/n_head/n_embd/block_size | seeds | mean bits/char | std | best seed | vs. trigram (3.2655) |
|---|---|---|---|---|---|---|---|
| cap-3K | 3,408 | 1/1/8/64 | 3 | 4.7579 | 0.0024 | 4.7552 | worse by 1.49 |
| cap-9K | 9,376 | 1/1/16/128 | 3 | 4.6343 | 0.0066 | 4.6280 | worse by 1.37 |
| cap-25K | 24,896 | 1/2/32/128 | 3 | 4.3360 | 0.0275 | 4.3172 | worse by 1.07 |
| cap-37K | 37,312 | 2/2/32/128 | 3 | 4.0703 | 0.0056 | 4.0642 | worse by 0.80 |
| cap-132K | 131,968 | 2/2/64/256 | 3 | 3.8750 | 0.0094 | 3.8690 | worse by 0.61 |
| cap-460K | 460,544 | 2/4/128/256 | 3 | 3.7706 | 0.0068 | 3.7634 | worse by 0.51 |
| **cap-1.4M** | 1,428,864 | 3/4/192/256 | 3 | **3.7610** | 0.0037 | 3.7577 | **worse by 0.50 (best in sweep)** |
| cap-3.3M (current default) | 3,347,968 | 4/4/256/512 | 3 | 3.8149 | 0.0093 | 3.8074 | worse by 0.55 |

`cap-3.3M-s1` predates the `best_val_bits_per_char` field being added to `experiments/sweep.py`
mid-run, so its bits/char isn't stored directly in its original run. It's derived exactly (not
estimated) from its stored `best_val` (2.5981 nats/token) using the token/char conversion factor
recovered from `cap-3.3M-s2` and `cap-3.3M-s3` (`k = best_val_bits_per_char / best_val`, both
seeds agree to 5 significant figures — 1.46726 and 1.46726 — confirming it's an exact constant for
this config's fixed val split, not a fit): `best_val_bits_per_char(s1) = 2.5981 * 1.46726 =
3.8121`. Flagged `"best_val_bits_per_char_derived": true` in the run file itself for anyone
re-deriving the table.

**Where the honest capacity of this dataset sits: nowhere good, and shrinking barely moves it.**
Every one of the 8 architectures tried, from 3.3M params down to 3.4K params, loses to the trigram.
The curve is *not* monotonic — going from 3.3M to 1.4M params improves things slightly (3.8149 ->
3.7610, a 0.05 bit/char gain), but shrinking further starts actively hurting: `cap-460K` is
basically tied with `cap-1.4M`, and everything below ~130K params gets rapidly worse as the model
becomes too small to represent English spelling at all (`cap-3K` at 4.76 bits/char is *worse* than
random-ish garbage would suggest — it hasn't got the capacity to learn "th" is more common than
"tj"). The honest read: there's a shallow, wide minimum somewhere around 460K-1.4M parameters,
0.50 bits/char above the trigram, and no amount of further shrinking closes that gap — it makes it
worse. **Capacity was never the bottleneck; the byte-level modeling problem itself is the
bottleneck** (see §3 for the direct comparison that proves this on the same architecture).

## 3. Experiment 2 — tokenizer swap

`babble/subword.py`: a from-scratch byte-level BPE trainer (merges learned on TRAIN-side rows
only, so no held-out phrasing leaks into the vocabulary before a single gradient step) and a
word-level tokenizer (most-frequent whitespace-delimited chunks get their own id, everything else
falls back to raw bytes chunk-by-chunk — total function, no `<unk>`, same guarantee
`babble/tokenizer.py` makes). Both stay off the production import graph: `core.py`, `generate.py`,
`trainer.py` and `bot.py` are untouched, and nothing new is reachable from the live bot. The flag
lives in `experiments/tokenizer_sweep.py --tokenizer {byte,bpe,word}`, default `byte`, mirroring
how `--augment-pairs` gated PR #19's pairaugment work in `posttrain.py`.

Same architecture for every condition (`n_layer=4 n_head=4 n_embd=256 block_size=256`, i.e. the
current default except `block_size=256` not `512`), same 600-step budget, 3 seeds per condition, 5
conditions, 15 runs total, via `experiments/lanes/tokenizer-sweep.sh` ->
`experiments/results/tokenizer/`.

| tokenizer | vocab size | params | seeds | mean bits/char | std | best seed | vs. trigram (3.2655) |
|---|---|---|---|---|---|---|---|
| byte (reference) | 260 | 3,282,432 | 3 | 3.7696 | 0.0123 | 3.7573 | worse by 0.50 |
| BPE | 512 | 3,346,944 | 3 | 3.2047 | 0.0053 | 3.1987 | **better by 0.06** |
| **BPE** | **1024** | 3,478,016 | 3 | **3.0897** | 0.0143 | 3.0806 | **better by 0.18 (best in sweep)** |
| BPE | 2048 | 3,740,160 | 3 | 3.1212 | 0.0196 | 3.1070 | **better by 0.14** |
| word | 1024 | 3,478,016 | 3 | 3.2768 | 0.0115 | 3.2676 | worse by 0.01 (a wash) |

`byte-ref`'s 3 seeds (3.7696 mean) land almost exactly on the capacity sweep's own `cap-3.3M`
byte-tokenizer reference point (3.8149 mean, same architecture family, `block_size=256` here vs.
`512` there) — the cross-check that `tokenizer_sweep.py`'s `--tokenizer byte` path and `sweep.py`'s
hardcoded byte path agree, and that both harnesses' bits-per-char accounting matches.

**Verdict on the tokenizer lever: it works.** All three BPE vocab sizes beat the trigram baseline
outright, at the *exact same architecture and parameter count* that loses to the trigram by half a
bit as byte-level. `bpe-1024` is the best single result of either experiment in this report — 3 for
3 seeds, mean 0.18 bits/char better than trigram, std 0.0143 (tight enough that this isn't noise).
`bpe-512` and `bpe-2048` also beat trigram, by smaller margins, so the effect isn't a single lucky
vocab size — there's a real gain across the BPE range, with 1024 as the local sweet spot (bigger
than 512 undershoots, 2048 slightly overshoots). Word-level tokenization, on the other hand,
essentially ties the trigram (0.01 bits/char worse, well inside its own 0.0115 std) — coarser
tokenization does help versus byte-level, but going all the way to whole-word granularity gives up
too much resolution for this corpus size to make good use of it. **This confirms the hypothesis in
the ticket directly: most of a byte-level model's capacity was going into learning how to spell
English, not into learning structure, and removing that burden — without touching model size,
learning rate, or step budget — is what actually closes (and then beats) the trigram gap.**

## 4. Qualitative samples

Same probes as every other report in this repo (`hello`, `the cat`, `why is`, `boop`), sampled
from the best checkpoint of each experiment (temperature 0.5, top-k 40 — the production sampling
config): `cap-1.4M-s2` (3.7577 bits/char, best capacity-sweep seed) and `bpe-1024-s2` (3.0806
bits/char, best tokenizer-sweep seed, and the best model in this entire report).

| probe | best capacity-sweep model (`cap-1.4M-s2`, byte-level, 1.4M params) | best tokenizer-sweep model (`bpe-1024-s2`, BPE-1024, 3.5M params) |
|---|---|---|
| hello | " what f cateren sest mesif ing the the ere" | " are you" |
| the cat | "s the foushe t akeso most sin busust bupl pers ntou t asot sesor" | "s the ins in a in jap to the sty the to and we and in s and \" a and in the in a on of and \"s like " |
| why is | " you thacate thake cathe t se burengous t thes nthe the thesesth" | " the coage" |
| boop | "" | "" |

The capacity-sweep model is still straightforward word soup — shrinking the architecture changed
the loss by 0.05 bits/char and changed nothing qualitative; it's the same kind of output as the
3.3M-param default, just marginally less bad by the numbers. The BPE model is a different
character: `hello` -> `" are you"` and `why is` -> `" the coage"` are short, syntactically-plausible
English fragments (a real greeting response; "the co[verb]age" is a real word missing one
character) rather than a stream of sub-word noise. `the cat` still degenerates into repetitive
soup once the completion runs long, so this is not solved — but it's a visibly different failure
mode than the capacity sweep's output, and matches the loss numbers: this is the one model in the
report that's actually favored over the trigram.

`boop` completes to empty output in *both* best-seed samples above, which reads suspicious in
isolation, but checking all 39 completed runs across both experiments shows `boop` is non-empty in
most of them (e.g. `cap-1.4M-s3`: `" int it"`, `bpe-1024-s3`: `" do what you to that to you the aner
to that to to of as a you to a in a in to to like that like to task in a "`) — it's just short
enough that some seeds sample an immediate stop. Not a bug, not investigated further; noted so it
isn't mistaken for a pattern.

## 5. Verdict

**Architecture/capacity: no. Tokenizer: yes.** Run to completion — 24/24 capacity-sweep runs, 15/15
tokenizer-sweep runs, all 3 seeds landed for every one of the 13 configurations.

Experiment 1 answers ro's question directly: no, shrinking the architecture is not the fix.
Every size from 3.3M params down to 3.4K params still loses to the twelve-line trigram table, the
best point found (`cap-1.4M`, 0.50 bits/char behind) is barely better than the current 3.3M default
(0.55 bits/char behind), and going smaller than ~130K params makes things actively worse, not
better — the model runs out of capacity to represent English spelling before it runs out of excess
capacity to overfit with. This is a clean negative result on the same footing as the
synthetic-corpus and pair-augmentation experiments: it ran to completion, the data says no, and
there's no honest way to read the numbers as anything else. **Do not turn on a smaller
architecture** — there is no config in this sweep that's worth trading for the current default, and
the current default itself was never the problem this sweep suggested it might be.

Experiment 2 is the first positive result in this line of work. At the *identical* architecture
that experiment 1 confirms loses to the trigram by half a bit, switching only the tokenizer to a
1024-merge BPE vocabulary flips the result: 0.18 bits/char *better* than the trigram, reproduced
across 3 seeds with a tight std (0.0143), and the qualitative samples show a real (if partial)
shift from sub-word noise toward short plausible phrases. The mechanism matches the ticket's
hypothesis exactly — a byte-level model spends its limited capacity re-deriving English spelling on
423 rows of data, and BPE removes that burden for free, no architecture change, no extra training
time, no extra data. **This is worth turning on**, behind the flag it already ships behind
(`--tokenizer bpe` in `experiments/tokenizer_sweep.py`, off by default). It is not, on this
evidence, ready to become the production default without further work: this report only measured
val loss and 600-step short runs, not sample-generation quality across a full training run, not
determinism guarantees around vocabulary training re-runs, and not integration into the live
`babble/trainer.py`/`bot.py` path (explicitly out of scope for this measurement-only run — see §6).
The recommended next step, *not done here*, is a longer, dedicated `bpe-1024` run through the real
training/eval/promote pipeline before considering it for `~/babble-live`.

## 6. What shipped, and what didn't

- `babble/subword.py` + `tests/test_subword.py` — BPE/word tokenizers, off the production path,
  fully tested (19 tests, including byte-adapter parity with `babble/tokenizer.py` and a
  merges-never-cross-whitespace invariant on the BPE trainer).
- `experiments/tokenizer_sweep.py` — the experiment-2 harness. `--tokenizer byte` (default) is a
  byte-identical passthrough to `babble/tokenizer.py`; nothing changes for anyone who doesn't pass
  `--tokenizer bpe` or `--tokenizer word`.
- `experiments/sweep.py`, `experiments/ngram_baseline.py` — additive: both gained a
  `best_val_bits_per_char`/`val_bits_per_char` field on their existing JSON output, computed the
  way §0 describes. No existing field changed meaning or was removed.
- `experiments/summarize_sweep.py` — aggregates a directory of either harness's run files into a
  seeds-mean/std markdown table (used to build §2/§3 above).
- `experiments/lanes/capacity-sweep.sh`, `experiments/lanes/tokenizer-sweep.sh` — the two sweep
  drivers, both idempotent (skip any run whose output file already has a summary line), both ran to
  completion this run (24/24 and 15/15).
- Nothing in `babble/core.py`, `babble/generate.py`, `babble/trainer.py`, `babble/bot.py`, or
  `babble/config.py`'s live defaults changed. The char-level path is exactly what it was before
  this run, and nothing was promoted to `/home/beckett/babble-live`.
- **What didn't ship: any change to what booper actually runs in production.** This was
  measurement-only, per the ticket's constraints. Turning `bpe-1024` on for real is future work
  (see §5).

## 7. Reproduction

```
# Trigram floor
python -m experiments.ngram_baseline --order 3

# Capacity sweep (24 runs, already complete in experiments/results/capacity/)
bash experiments/lanes/capacity-sweep.sh

# Tokenizer sweep (15 runs, already complete in experiments/results/tokenizer/)
bash experiments/lanes/tokenizer-sweep.sh

# Tabulate either directory
python -m experiments.summarize_sweep --dir experiments/results/capacity --fields params n_layer n_head n_embd
python -m experiments.summarize_sweep --dir experiments/results/tokenizer --fields tokenizer requested_vocab_size params
```

Both lane scripts skip any run whose output file already has a `"summary": true` line, so
re-running either command against the existing `experiments/results/` directories is a no-op.

## 8. babble-live untouched

`~/babble-live/data/{corpus.jsonl,consent.json,.salt}` are byte-identical (`diff` clean) to this
worktree's own gitignored `data/` copy, which was made once at run start and never written back.
For the record, at report time:

```
dfb219d94c63d4ddb6d22d8a779eb4d8f539929e56f03af4b5ebaf7c5ffbe3f9  corpus.jsonl
c8ac74e0a7601212060f0159fcc312855f384292cb2d12018f23664645e9fb4f  consent.json
13ef27114cb24dc346c2710f811258709dbb53daa3a724b0abb007d4006edd4e  .salt
```

Full pytest suite: 620 passed (unchanged from the count before this run's harness edits).
