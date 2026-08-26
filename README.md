# babble

A language model that starts as **pure noise** and learns to talk from a corpus
people write by talking to it in Discord.

> "what if we make a model that learns how to talk by people interacting with it
> on discord. So you ping it, then you get a replied response. If you reply to
> that you can correct it with the response you wanted whether it be a gif and
> whatnot."
>
> "No but the from scratch is the most fun. Because it'll be completely random
> and we train on checkpoints in a consistent ultra efficient loop that doesn't
> make the computer unusable"

> "in general right now we are collecting english corpus and overfitting it to a
> pair. which is fine for now but ideally itd just be a pool of unlabelled data
> to be picked at"

There is no pretrained base and no scraped chat history. `babble` is ~3.3M
randomly initialised parameters, and the only data it will ever see is the
**corpus people fill by talking to it** — plain unlabelled text, one message at
a time, every word of it given deliberately by somebody who opted in.

It used to train on `(prompt, rejected, chosen)` correction *pairs*, overfitting
each prompt to its one right answer. It doesn't any more. Corrections are still
captured and still published, they are just no longer the objective: the model
does plain next-token prediction over the corpus, and what it gives you back is
a **continuation** of what you said, not an answer to it. See
[the pivot](#the-pivot-from-pairs-to-a-corpus).

## Set your expectations

**It will babble for a very long time, and it may never become coherent.**

That is not a bug to be fixed, it is the thing you are watching. A byte-level
model learning English from a few hundred corrections is going to produce
garbage for a long, long time. The first outputs look like this:

```
'hello' -> '\x0fsKR�1���dE�/��8\x1d�EE�ح'
```

The loss curve and the per-checkpoint sample are the show. If you want a chatbot
that works, use a chatbot that works.

## Quickstart

No Discord token needed for any of this.

```bash
git clone https://github.com/kowo-co/babble && cd babble
uv venv && uv pip install -e ".[dev]"      # or: python -m venv .venv && pip install -e ".[dev]"
source .venv/bin/activate

babble fake-data                # made-up rows in both stores, to chew on
babble train --force            # STAGE 1: random init -> the human corpus -> latest.pt
babble synth-generate           # postulate prompts for reply-shaped corpus rows -> synthetic_pairs.jsonl
babble post-train --force       # STAGE 2: fine-tune latest.pt on the correction pairs
babble sample --prompt hello    # continue a prefix from the newest checkpoint
babble curve                    # the loss curve, as a picture
babble summary                  # step, loss, checkpoints, consent, row counts
babble backfill-corpus          # flatten old pairs into the corpus (runs itself too)
babble export                   # build the HuggingFace dataset directory
pytest                          # 500+ tests, none of which need a token
```

`babble train` is the one command that pretrains the model — see
[Pretraining](#pretraining) for what it trains on, when it fires on its own,
and when you need `--force`. After that, a short supervised pass on the
correction pairs is described under
[Post-training on the correction pairs](#post-training-on-the-correction-pairs).

`uv` pulls the CPU-only build of torch automatically (see `[tool.uv.sources]` in
`pyproject.toml`) — no multi-gigabyte CUDA wheels. With plain pip, use
`pip install torch --index-url https://download.pytorch.org/whl/cpu` first.

Delete `data/` before going live, or the fake rows will look like things real
people said.

## How the loop works

1. **You ping it.** `@babble hey` — it replies with whatever its current weights
   produce, continuing from what you wrote.
2. **What you wrote goes into the corpus.** That is the whole collection
   mechanism. Not the reply — the model's own output is never stored; a corpus
   of what a random model emitted is not a corpus of human writing.
3. **Every reply carries a footer** telling you how to grade it, if you want to.
4. **You react 👍** — a weak "that was fine", still recorded as a correction-side
   signal.
5. **Or you reply starting with `>>`** — filed as a correction *and* added to the
   corpus as your own writing:

   ```
   >> hey, what's up
   ```

   Text, an emoji, a URL, a gif link, an uploaded image: whatever follows the
   marker is stored raw.

Step 2 is the change. You do not have to teach it anything for your message to
count — talking to it is enough, which is the point:

> "So people can just talk to booper to collect that corpus"

## The pivot: from pairs to a corpus

The old objective paired every prompt with one right answer and trained the
model to reproduce that answer, with the prompt tokens masked out of the loss
and corrections weighted above 👍 rows. It worked, in the narrow sense that the
loss went down. It also meant a dozen people's worth of real English was being
squeezed through a dozen `(prompt → answer)` slots and overfitted to them.

What is collected now is a **pool of unlabelled text**, and what is trained is
plain next-token prediction over all of it:

| | before | now |
| --- | --- | --- |
| what is stored | `(prompt, rejected, chosen)` triples | one row per message, plus the triples |
| what is trained | `chosen`, prompt masked out | every token of every corpus row |
| per-row weights | corrections ×3, 👍 ×0.25 | none — every row counts once |
| what the bot does | answers a prompt | continues what you wrote |
| the probe in the feed | prompt → output, vs the expected answer | a prefix → what it continues with |

Nothing was thrown away. The correction pairs are still captured, still stored
in `data/interactions.jsonl`, and still published — as their own config on the
Hub. They are flattened into the corpus so the text people already gave is not
stranded in a format nothing reads any more. **That migration runs by itself**,
at the top of every training cycle and once at bot startup, so upgrading a box
that already has data needs no command: it is idempotent, it dedupes on the
content-addressed row id, and it only logs when it actually added something.
`babble backfill-corpus` runs the same migration by hand if you want to watch it.

**The model never sees a `<sep>` any more**, so nothing at inference time may put
one in front of it. `<bos> text` is what it trained on and `<bos> text` is what
the bot, the CLI and the checkpoint probe all generate from. The paired layout
still exists in `tokenizer.py` for *scoring* a correction pair, which is a
meaningful thing to ask; it is just not a thing the model is ever fed.

### Why corrections need the `>>` marker

Without it, *every* reply to one of the bot's messages was stored as the answer
it should have given — so "lol", "wrong" and "what even is that" all went into
the pairs as lessons. That is a set of pairs full of things nobody meant to
teach it.

The marker matters less than it used to, because an unmarked reply now goes into
the corpus anyway as what it actually is: a message someone sent the bot. What
the marker still decides is whether it is *also* filed as a correction — a claim
about what the right answer was. That claim should be deliberate.

So a correction has to say it is one:

- A reply **beginning with `>>`** is a correction. The marker, plus one space
  after it, is stripped before anything is stored — it never reaches the corpus,
  the dataset, or the model.
- A reply **without** it is just a message to the bot. It gets answered like any
  other, and the decision not to learn from it is recorded as `capture.unmarked`
  rather than silently swallowed. (Not `bot.dropped` — nothing was dropped; the
  reply was answered.)
- `>>` **on its own** is rejected, with a short reply showing the format.
- **👍 reactions are unaffected.** This is a reply-path rule only.

The marker is `CORRECTION_MARKER` in `babble/config.py` — change it in that one
place and the footer, the help text, the consent notice and the parser all
follow.

Each correction is kept as a triple:

| what was said to it | what it answered | what it should have said |
| --- | --- | --- |
| `prompt` | `rejected` | `chosen` |

A 👍 is the same shape with `rejected: null` — you are saying the answer it gave
*was* the right one.

Because tokenisation is byte-level, a gif correction works exactly like a text
one. A word-level vocabulary would have to fall back to `<unk>` on
`https://tenor.com/view/...`; bytes just spell it out.

## Consent

Nothing about anyone is stored until they have explicitly opted in.

The **first** time you ping the bot you get the notice instead of an answer: it
explains that the messages you send it are stored, used to train it, and
**published to a public HuggingFace dataset**. You then choose.

```
!babble accept     opt in
!babble decline    no thanks — it still replies, it just keeps nothing
!babble all        in THIS channel, collect everything you say, not just pings
!babble pings      undo that, here, immediately
!babble forget     opt out and delete everything of yours it has kept
!babble consent    show the notice and your current setting
!babble status     how training is going
```

### What "opted in" covers by default

**Only what you address to the bot**: an @mention, a reply to one of its
messages, or a DM. Everything else you say in a channel is yours, is never read,
and never reaches the corpus. That boundary is the default because it is the one
people actually have in mind when they say yes to a bot.

### Widening it, one person and one channel at a time

`!babble all`, run **in a channel**, collects every message *you* send *in that
channel* from then on — not just the ones aimed at the bot. It exists for the
real case it was asked for: a dedicated bot-testing channel where somebody wants
everything they type to feed the corpus.

- **Scoped to you.** It never widens collection for anyone else in that channel.
- **Scoped to that one channel.** It is not server-wide and does not follow you
  elsewhere. Two channels means running it twice.
- **Off by default**, always, for everyone.
- **`!babble pings` turns it off**, and that takes effect on your very next
  message. Rows already stored stay until you `!babble forget`.
- It requires the ordinary grant first. Running it without having opted in shows
  you the notice instead of widening anything.

Ambient messages collected this way are stored with `source: "ambient"` and the
bot never replies to them.

One deliberate exception to "everything you say": `!babble …` commands are never
collected, in a widened channel or anywhere else. They are addressed to the bot's
dispatcher rather than written to be read, and a corpus full of `!babble status`
is a corpus of nothing.

### Two grants, tracked separately

Collecting the messages you send the bot is broader than collecting only your
corrections, so the older agreement does not silently stretch to cover it.
Consent is recorded per **scope**:

| scope | covers | who has it |
| --- | --- | --- |
| `corrections` | the `(prompt, rejected, chosen)` pairs | anyone who ever opted in, including under the old notice |
| `corpus` | the messages you send the bot | only people who answered the **new** notice |

A `consent.json` written before the corpus existed loads as `corrections:
granted, corpus: unknown` — so that person keeps every right the old notice gave
them, their existing correction rows stand, and **none of their ordinary
messages are collected** until they see the new notice and accept it again. They
get asked once, on their next ping, and the bot keeps answering them meanwhile.
A legacy *"no"* carries across to both scopes: a refusal never needs re-asking to
stay a refusal.

**Which grant governs a corpus row is decided by where its text came from**, not
by which file it lives in. A row whose `source` is `prompt` or `correction` was
flattened out of a pair collected under the corrections notice, so the
`corrections` grant governs it: that text was already collected and published
under that notice, and re-filing it is the same words in a second index rather
than a new collection. A row whose `source` is `mention`, `reply`, `dm` or
`ambient` is the genuinely broader thing the corpus notice describes, so it needs
the `corpus` grant. An unrecognised source is held to the stricter of the two.

That table lives in `consent.SCOPE_BY_SOURCE`, and `consent.CorpusConsent` is the
single gate the trainer, the summary, the export and the migration all ask. They
used to each apply their own rule, which is how the pivot shipped with a
migration that wrote rows the trainer then refused to train on. An explicit
`declined`/`withdrawn` on `corpus` drops a person's rows whatever the corrections
grant says: a no is stronger than a yes, in every scope and from every source.

### The rules, all enforced in code and covered by tests

- **Fail closed.** No consent record means no data. Silence is not consent, a
  missing file is not consent, a corrupt file is not consent.
- **Checked at capture time**, not just at reply time — and again at training
  time, and again at export time.
- **Everyone in a correction must have opted in.** If Alice asks and Bob
  corrects, the pair needs both, because Alice's message gets published too. A
  corpus row has one author and needs exactly that person's grant.
- **Withdrawal is retroactive.** `!babble forget` deletes your rows from **both**
  stores, drops your pending exchanges, clears any channels you widened, and
  takes you out of every future export.
- **Raw Discord ids never reach the dataset.** They exist only in two local
  operational files (`data/consent.json`, `data/exchanges.json`). The stores, the
  logs and the export contain only salted hashes like `u_9f2c4a…`. Usernames are
  never read at all. Channel ids are kept raw in `consent.json` only, because
  scope cannot be enforced without them — a channel id names a room, not a
  person, and it is never exported.

The salt lives in `data/.salt` (generated once) or `BABBLE_HASH_SALT`. **Never
change it** — `!babble forget` finds your rows by re-deriving your hash.

## Content filter

**Be honest about what this is: a speed bump, not a guarantee.** A word list
with some normalisation catches copy-pasted slurs and the lazy attempts to get
around them — dots or spaces jammed between letters, a leetspeak substitution,
an accent. It will not stop someone determined to get past it. Treat it as one
layer, not the whole plan.

If the prompt, the correction, or the model's own generated output matches a
blocked term, **the whole row is rejected** — never stored, never trained on,
never published. This is checked in the same three places consent is: at
capture time, again at training time, and again at export time, so a row
stored before a term was added can never sneak through once the list is
extended. The model's own output is checked at the one place every generation
passes through, before it is ever sent to Discord.

The list lives in `babble/blocklist.py`'s companion file, `babble/blocklist.txt`
— a small, starter set across a few categories, deliberately not an attempt at
an exhaustive one (no such list exists). One term per line, `#` for comments.
Edit it directly, or point `BABBLE_BLOCKLIST_PATH` at your own file entirely.

Matching happens on a **normalised** form: lowercased, accents stripped,
common leetspeak folded back to letters, repeated characters collapsed, and
separator punctuation (dots, dashes, underscores, zero-width characters, and
letters spaced one at a time) glued back together — so `s.l.u.r`, `sluuur` and
`s l u r` all fold to the same form a plainly-typed term would. Matching still
requires the *whole word*, not a substring, so ordinary words that happen to
contain a blocked fragment (`class`, `assassin`) are not false-positived.

A rejection is logged — the reason and a content hash, never the text itself —
and the person is told briefly that it wasn't accepted, without the term being
repeated back to them.

```bash
babble rescan-blocklist   # purge stored rows that now match the current list
```

Run this after extending the list, so history gets cleaned up along with it.

## Pretraining

**One corpus, one command.** babble trains on nothing but the corpus people
actually gave it — no external text, no frozen base to continue from. Every
run of `babble train` starts from *random init* and trains on the consented
human rows only, and writes `checkpoints/latest.pt` (what the bot serves).
There used to be a two-stage design here — a frozen base pretrained on an
external word list and TinyStories, then a cheap "voice pass" from that base —
and it is gone: it put text into the model that nobody in Discord ever wrote,
which is not what this project is for.

**Training is opt-in and never ambient.** Nothing starts training on its own —
not the bot on boot, not any implicit scheduler, and never a continuous loop
(an earlier continuous `babble train --loop` was retired outright — it burned
CPU for nothing, running to step 3,200 with the loss flat since step 2,000).
`babble train` runs once and returns:

```bash
babble train --force        # run right now, regardless of the trigger
babble train --force --steps 200 --patience 5   # override the ceiling / patience
babble train-status         # rows since the last run, whether the trigger is due
```

**A trigger, not a loop.** `babble train` (no `--force`) is a no-op unless the
corpus has grown by `BABBLE_TRAIN_TRIGGER_ROWS` (default 100) new rows since
the last run — it prints why and stops rather than silently retraining every
time someone runs it. The running bot watches for that crossing itself and
launches `babble train` as a detached, low-priority subprocess, then
[hot-reloads](#watching-it) the new `latest.pt` on its own — this is the
*only* automatic path, and it needs no `base.pt` and downloads nothing. The
last-trained row count is persisted in `checkpoints/train_state.json`, so a
restart never re-fires. Set the threshold to 0 to run by hand only.

**`--steps` is a ceiling, not a target.** On a small corpus the model overfits
long before the step budget is spent — val loss bottoms out, then climbs. So
`train()` tracks val loss at every checkpoint interval and, **at the end of the
run**, writes whichever step had the *lowest* val loss to `latest.pt` — never
just whatever the last step happened to produce.

**Early stopping is noise-aware.** On a corpus this size the val estimate
itself is noisy: hold a checkpoint fixed and recompute val over thousands of
resampled 20% holdouts and the spread is ~0.05 nats std
(`experiments/val_noise.py` measures it; an early version of this stop once
killed a run over a 0.075 wobble — within that band — while train loss was
still falling fast). So "failed to improve" is not evidence, and patience is
gated twice: it cannot fire before `BABBLE_TRAIN_MIN_STEPS` (default 600),
and a checkpoint only burns patience when val exceeds the best seen by more
than `BABBLE_TRAIN_STALL_MARGIN` (default 0.05, the measured band; 0 restores
the old hair-trigger). After `BABBLE_TRAIN_PATIENCE` (default 3) such stalls
the run stops. `train.stop` in `logs/babble.log` names the winning step, its
val loss, and how many steps ran versus the ceiling. Best-checkpoint
selection and the early stop are both no-ops without a held-out validation
set (too little data to spare any) — see [Validation](#validation) below, in
which case every checkpoint interval is simply written in turn, same as
before.

**Every run starts fresh.** There is no base and nothing to resume: each call
to `babble train` builds a brand-new random-init model and trains it on
whatever the corpus holds at that moment, so nothing compounds across reruns
and the human corpus is always the only thing the weights reflect.

**It is safe to kill at any moment.** Checkpoints are written to a temp file and
renamed, so `kill -9` mid-write leaves the previous one intact. `SIGINT`/`SIGTERM`
finish the current step, checkpoint, and exit cleanly.

Every checkpoint appends `{step, loss, sample}` to `checkpoints/loss.jsonl` and
prints it, alongside the worst single row and the held-out validation loss:

```
step     100 | loss   0.0263 | worst  1.012 | val   0.0891 | [trained] 'hey there' -> ' how are you'
```

**The objective is plain next-token prediction over the corpus.** Every token of
every row is a target — there is no prompt to mask off, because a corpus row is
one piece of writing rather than a question and its answer. Every row counts
once: no per-row weights, because there is no "chosen" answer left to weight.

A row longer than `BABBLE_BLOCK_SIZE` is split into consecutive examples rather
than truncated, so a long message contributes all of itself. Only the last chunk
gets an `<eos>` — that token means "the text ended here", and it only actually
did at the end.

The correction pairs — `chosen`, `rejected` and the weights — are stored and
published but nothing reads them at training time any more. See
[what "RL harder" actually means here](#what-rl-harder-actually-means-here).

> **Geometry.** `block_size` is **512** and `max_new_tokens` is **256**
> (`BABBLE_BLOCK_SIZE` / `BABBLE_MAX_NEW_TOKENS`). `block_size` is the byte
> context window; raising it grows the learned positional-embedding table, so
> it invalidates every checkpoint trained at the old size — harmless here since
> every run starts from random init anyway, so the next `babble train` just
> produces a fresh checkpoint at the new size.

## Post-training on the correction pairs

Pretraining teaches the model to *continue* plain text — there is no
prompt/response boundary in that objective, so nothing about it teaches the
model to *answer* a question. This stage does: a short supervised pass, run
**after** pretraining, that fine-tunes the pretrained checkpoint on the stored
`(prompt, chosen)` correction pairs, laid out as `<bos> prompt <sep> response
<eos>` (`tokenizer.build_example` — the same pair layout `generate.py` already
uses to score a correction against its rejected answer, reused rather than
inventing a second format).

```bash
babble post-train --force   # fine-tune the pretrained checkpoint on the pairs -> latest.pt
babble post-status          # pairs since the last post-train, whether the trigger is due
```

**Set your expectations here too.** There are only a few dozen correction pairs
against a ~3.3M-parameter model that just learned characters from a few
thousand characters of corpus. It will *memorise* those pairs and generalise to
approximately nothing — and measured on the live corpus, the original settings
did real damage on the way there: 38 pairs fine-tuned at the pretrain LR
shipped a checkpoint whose held-out corpus loss was **+1.15 nats worse** than
the pretrain it started from
([the revamp report](docs/reports/PIPELINE_REVAMP_2026-08-20.md) has the numbers). Note
also that the bot *serves* plain continuations, not the `<sep>` pair layout —
so what post-train teaches is never directly exercised at inference, and the
damage to the continuation ability is the whole effect the served bot sees.

**Four guardrails now stand between a fine-tune and the served bot**, each
config-flippable and each chosen from the measured grid in
`experiments/post_grid.py`:

1. **Its own learning rate** — `BABBLE_POST_LEARNING_RATE` (default `1e-4`).
   The historical pretrain LR (1e-3) tore through the weights in a handful
   of steps at this pair count.
2. **Rehearsal** — `BABBLE_POST_REHEARSAL` (default `0.5`): that fraction of
   every post-train batch is plain corpus text under the pretrain objective,
   so the fine-tune cannot drift off the corpus distribution unopposed.
3. **A pair floor** — `BABBLE_POST_MIN_PAIRS` (default `100`): below this
   many trainable pairs the run refuses to start and says so; `--force`
   overrides for an explicit experiment.
4. **A promotion gate** — `BABBLE_POST_GATE_MARGIN` (default `0.05`, the
   measured val-noise band): after training, the candidate is scored against
   the pretrain snapshot on the held-out corpus split. Worse by more than the
   margin → `latest.pt` is left untouched and the run reports itself
   `gated`. Nothing writes `latest.pt` mid-run any more either, so a
   half-finished fine-tune can never be what the bot serves. A negative
   margin disables the gate.

**The `rejected` side is captured, not trained on.** Every correction still
stores what the bot got wrong alongside what it should have said, and both are
still published. But the objective here is supervised fine-tuning on the
chosen answer only — this is not preference optimisation, and there is no DPO
or RL anywhere in this path.

**The pretrained/post-trained split.** The same discipline the old base/voice
split used: the first time a post-train ever runs, it snapshots whatever is
currently in `checkpoints/latest.pt` (the pretrain output) as
`checkpoints/pretrained.pt`. Every post-train — including every rerun — starts
from that snapshot, never from a previous post-train's own weights, so nothing
compounds and a post-train can always be redone from a clean pretrain. This
matters more here than it did for the voice pass: the pair set is tiny, so a
post-train that kept fine-tuning on top of its own previous output would drift
fast. Unlike the old base/voice split, the snapshot is not frozen forever: each
post-train checks whether `latest.pt` still holds exactly what the *last*
post-train wrote there (a content hash recorded in `checkpoints/post_state.json`
settles it), and retakes the snapshot when it does not — which is exactly what
happens when a fresh `babble train` (by hand, or the bot's own +N-row trigger)
has landed a new pretrain since. A post-train never silently fine-tunes a stale
pretrain, and never silently throws away a newer one.

**Best-val checkpoint and early stopping**, same as pretraining: `--steps` is a
ceiling, not a target. The post-train tracks val loss at every checkpoint
interval and writes whichever step had the lowest val loss to `latest.pt`, and
stops early once `BABBLE_POST_PATIENCE` (default 3) checkpoint intervals in a
row fail to beat that best. `post.done` in `logs/babble.log` names the winning
step and its val loss. Both are no-ops without a held-out set (too few pairs to
spare any), in which case every checkpoint interval is simply written in turn.
A non-positive `--steps` trains nothing and is rejected as a no-op before it
touches the pretrained snapshot or consumes the trigger.

**A trigger, not a loop.** Post-train re-fires every `BABBLE_POST_TRIGGER_PAIRS`
(default 10) *new* correction pairs since the last post-train, or on demand
with `--force`. The last-trained pair count is persisted in
`checkpoints/post_state.json`, so a restart never re-fires. Set the threshold
to 0 to run it by hand only. The running bot watches corrections for that
crossing itself and launches `babble post-train` as a detached, low-priority
subprocess — deferring if a pretrain it launched is still in flight, so a
post-train never starts against a pretrain that is about to be replaced. There
is no continuous loop here either — the old `train --loop` is gone for a
reason, and it does not come back for this stage.

**Zero pairs.** With no consented correction pairs yet, `post-train` is a
no-op that says so (`Nothing to post-train on: no consented correction
pairs.`) rather than crashing or writing a degenerate checkpoint.

`babble summary` reports the pair count and whether a post-train is due
alongside everything else.

## HF pretrain + Discord post-train (two-stage)

`pretrain_hf.py` (repo root) is a self-contained script -- no dependency on
this package being installed -- that pretrains booper's architecture on a
bounded, streamed slice of `openbmb/Ultra-FineWeb-L1` (an English web corpus)
on a real GPU, at a size (20-50M params, config-selectable) that the ~463-row
Discord corpus alone could never inform. It ships with three JSON presets
under `configs/pretrain/` and is meant to be handed to whoever owns the GPU:
`hf jobs uv run --flavor l4x1 --secrets HF_TOKEN <url-to-this-file> --
--config configs/pretrain/default.json --output-dir <path>`.

`babble post-train-from-checkpoint --checkpoint <ckpt.pt> --tokenizer
<tokenizer.json>` is stage 2 against whatever `pretrain_hf.py` produces:
the same guardrails as `post-train` above (own LR, rehearsal, pair floor,
best-val selection, promotion gate), generalised over the pretrain's own
tokenizer instead of hardcoding `babble.tokenizer`'s raw bytes.

Full writeup — dataset/model-size/tokenizer justification, measured
throughput, the exact SSH-facing command, cost estimates, and what "done"
looks like once a real checkpoint comes back — lives in
[HF_PRETRAIN_PIPELINE.md](docs/reports/HF_PRETRAIN_PIPELINE.md).

## Synthetic correction pairs

ro's ask: post-train has only ever had a few dozen human corrections to learn
from, and adding more one at a time is slow. Rather than inventing text
nobody wrote, `babble synth-generate` looks at the corpus that already
exists for rows that read like a *reply* or an *interjection* — his
example, `"well the visual shells were also just *fine* again i wasnt drawn
to *any* of them"` — and postulates the prompt each one was plausibly
answering.

**The response side is never rewritten.** A synthetic pair pairs a
*synthesized prompt* with the *verbatim* corpus row as the response. That is
the whole trick: voice replication is "to a tee" by construction, because the
response half is not synthesized at all — it is the same text a real person
already wrote, byte for byte. Only the prompt in front of it is new.

**Continuation cuts grow the response pool too.** Postulated prompts alone
turned out to be a measured null: they only ever synthesize the *prompt*
side, so the pool of response text — the half the loss is actually computed
over — never grew. `synth-generate` therefore also cuts every corpus row of
four words or more at word boundaries into `(prefix → rest)` pairs
(method-tagged `continuation_cut`, up to `--cuts` per row, default 2). Both
halves are byte-slices of the original row: nothing is invented, and the
pair teaches exactly the mapping the pair layout exists for — "given this
opening, produce the rest, in this voice". On the current corpus that is
~500 pairs against 47 human corrections. `--no-continuations` /
`--no-postulate` select either generator alone.

```bash
babble synth-generate      # scan the corpus -> data/synthetic_pairs.jsonl
babble synth-status        # synthetic vs human data counts
```

**Be honest about the method.** There is no LLM credential wired into this
repo — `babble/config.py` has no API-key setting, and the dependency list is
`torch` / `discord.py` / `huggingface_hub`, nothing that talks to a model API
— so "postulate" here does not mean asking one. It means a small set of
surface heuristics: an interjection at the front (`yeah`, `well`, `honestly`,
…), a continuation word (`also`, `again`, `still`), an emphasis marker
(`*word*`), a trailing `?`, each mapped to a template filled with a
stopword-stripped topic phrase pulled from the row itself
(`babble/synthetic.py`, `is_reactive` / `synthesize_prompt`). It is not
language understanding, and the postulated prompts read like it — clunky,
sometimes ungrammatical, always guessable as templated rather than typed. It
is enough to demonstrate the mechanism ro described; it is not enough to
call the prompts natural.

**Kept strictly apart from human corrections.** Every pair lives in its own
file, `data/synthetic_pairs.jsonl` (`Settings.synthetic_pairs_path`), and is
never appended to `interactions.jsonl` — there is no code path that mixes
the two files. `babble post-train` ignores `synthetic_pairs.jsonl` entirely
unless told `--include-synthetic`; generating synthetic pairs never changes
what a plain `babble post-train` trains on, and never makes a post-train
*due* — the +N-pair trigger counts human corrections only. Deleting the file
(or just never passing the flag) turns synthetic data off completely.

**Re-runnable.** Every pair is content-addressed against its source corpus
row, so running `synth-generate` again after the corpus has grown only adds
pairs for the rows that are new since the last run — it is safe to wire into
the same cadence as `babble train`, and re-running it twice in a row adds
nothing the second time.

**Consent, twice.** A synthetic pair is only ever built from a corpus row
that currently passes the same consent + blocklist gate the trainer applies,
and `--include-synthetic` re-checks that gate again at train time — so a
`!babble forget` that purges someone's corpus row also stops any synthetic
pair built from it from being trained on, the same belt-and-braces promise
`trainable_pairs` gives human corrections.

```bash
babble post-train --force --include-synthetic   # human + synthetic pairs -> latest.pt
```

## Synthetic corpus rows

The pair generators above feed post-train. The pretrain side has its own
corpus-internal expansion: `babble synth-corpus` recombines corpus phrasing
into *new rows* via an order-2 word-level Markov chain built over the
consented corpus (`babble/synthcorpus.py`).

**Why this counts as corpus-internal by construction.** Every emitted word
is a word somebody actually typed, spelled exactly as they typed it; every
consecutive word pair follows a transition observed in the corpus (the chain
only moves `(w1, w2) → w3` where `w1 w2 w3` appears verbatim in some row,
with order-1 backoff only from dead-end contexts); row starts are real row
openings and rows end where a real row ended. A generated row is a splice of
real phrasing — lowercase Discord cadence, slang, typos and all — in a new
but in-distribution order. Verbatim replays of real rows are rejected, so
every stored row is genuinely *new* text.

```bash
babble synth-corpus --count 400        # -> data/synthetic_corpus.jsonl
babble synth-corpus --rebuild          # regenerate from the current consented corpus
babble synth-status                    # counts, synthetic vs human
```

**Same separation discipline as the pairs.** Rows land in their own file
(`data/synthetic_corpus.jsonl`), each labelled `"synthetic": true` with the
method that made it; nothing is ever appended to `corpus.jsonl`. The trainer
mixes the file in by default (`BABBLE_TRAIN_SYNTHETIC=0` makes it ignore the
file entirely), and only ever into the **train side** — validation stays 100% real held-out
human rows, so the number a run is judged by can never be flattered by the
generator trying to improve it.

**The chain never sees held-out rows either.** Generation excludes the rows
the trainer's deterministic split holds out for validation
(`babble/valsplit.py`, the split's single torch-free definition), so a
synthetic row cannot splice val phrasing into the train side and quietly
deflate val loss. `--include-val-sources` restores the whole-corpus chain
for experiments that need it.

**And a stale file cannot re-open that leak.** The holdout is a slice of the
whole id population, so appending corpus rows migrates some existing rows
train → val — a file generated earlier could then contain splices of
now-held-out phrasing. Each generation records the exact source-id set in a
sidecar (`data/synthetic_corpus.meta.json`), and the trainer rebuilds the
file from the current corpus before mixing whenever that recorded set no
longer matches the current train side (a file with no sidecar, or one built
with `--include-val-sources`, is treated as stale too).

**Consent.** Rows are generated only from text passing the same consent +
blocklist gate the trainer applies. A spliced row has no single source row
to track a withdrawal against, so the supported way to honour one is
`babble synth-corpus --rebuild`, which regenerates the whole file from the
*current* consented corpus (withdrawal already purged the source rows by
then). The trainer re-checks the blocklist on every synthetic row at
training time.

## Why it babbled at loss 0.02

> **Written under the old paired objective.** The diagnosis is still exactly why
> a low mean loss can sit next to bad output, and the worst-row reporting it
> produced is still in the trainer — but the specific `response_loss` /
> `prompt_loss` pair below no longer exists, because there is no mask left to
> check. Kept because the reasoning is the useful part.

ro asked the obvious question: the trainer was reporting a train loss around
`0.02` on nineteen rows, which should mean the corpus is memorised, and the bot
still answered `hi` with `smeu.nnuccrl,`. A loss that low and output that bad
cannot both be telling the truth. Here is which one was lying, and how we know.

**It was not the prompt tokens.** The first guess is that the loss averages over
`<bos> prompt <sep> response <eos>` uniformly, so `0.02` is mostly the model
learning to parrot back the prompts. Measured on a 19-row corpus, that is not
what happens: response loss falls to `0.0001` while prompt loss *rises* from
`6.19` to `7.68` nats (independently re-run at the shipped model shape:
`0.0002` against `8.06`). The mask was already correct and the reported number
was already response-only. Both numbers are now logged separately
(`response_loss`, `prompt_loss`) so nobody has to take that on faith again —
and if `prompt_loss` ever starts falling, the mask has broken.

**The reported loss is a mean over tokens, and the mean is where the problem
hid.** Two separate averages were doing the hiding:

- *Across rows.* Train eighteen rows to convergence, add a nineteenth, and the
  new row sits at loss `1.01` while contributing 7 of 276 masked tokens — 2.5%
  of the denominator. A row can be completely unlearned and move the headline
  number by a rounding error. This is the normal state of this bot: every
  correction ro types is a fresh row in an otherwise-memorised corpus.
- *Across tokens within a row.* At a response loss of `0.027` the *worst single
  token* in the corpus was at loss `1.92` — the model was assigning the correct
  next byte about a 15% chance at that position, while the average said `0.027`.

So the checkpoint line now prints the worst row next to the mean, and
`train.checkpoint` logs `worst_row_loss` and `worst_row_text`. That is the
measurement that turns "loss 0.02 but it babbles" from a mystery into a number.

**Temperature 1.0 then cashed that uncertainty in.** A byte-level model has no
way back once it leaves the memorised path: one unlucky byte and the rest of the
response is off-distribution, which is exactly what character soup looks like.
Measured at response loss `0.027`, sampling the corpus back:

| decoding | exact reproductions |
| --- | --- |
| temperature 1.0, top_k 40 (old default) | 79% |
| temperature 0.6, top_k 40 | 91% |
| temperature 0.4, top_k 40 | 84% |
| greedy | 16/19 |

and every failure has the same shape — correct prefix, one wrong byte, then
noise: `hey! whati  up`, `heyo lidti atwethmuch`. On the fresh-correction corpus
above, temperature 1.0 produced `ghwhthhke :3hththh fhhhe` for `hi` — ro's
symptom exactly — while temperature 0.5 returned the memorised
`hey! what's up` on two draws out of five.

Re-run since on the **shipped** model shape (4 layers, 256 wide, 3.3M params),
nineteen rows, stopped at response loss `0.0076` — *lower* than the number that
prompted the question — four draws per prompt:

| decoding | exact reproductions |
| --- | --- |
| temperature 1.0, top_k 40 (old default) | 67/76 |
| temperature 0.5, top_k 40 | 76/76 |
| best-of-4 at 0.5 (shipped) | 76/76 |

Same signature on every miss: `prett inood, still learning`,
`ro did, mostly, and it shop`. So at *twice as good* a loss as the one in the
title, the old defaults still mangled one reply in eight, and the fix is not in
the optimiser.

The default is now **`0.5`** (`BABBLE_TEMPERATURE`). Not greedy: greedy is
deterministic, so when it is wrong it is wrong every single time and there is no
second draw to rescue it — in the same measurement greedy got stuck emitting
`hey! what's o nehreahoucop`. Which is what best-of-n is for.

**One honest caveat about the regression test.** The brief for this work expected
the memorisation gate to fail on the old code. It does not, and that is worth
saying rather than hiding: three short rows trained to convergence reach a loss
around `1e-5`, and at that point there is no residual uncertainty left for a hot
sampler to cash in — temperature 1.0 reproduces them perfectly too. The bug lives
at the *boundary*, on rows the model has partly learned, which is the permanent
condition of a bot that gets a new correction every day. So there are two gates:
one trains to convergence and guards the training path, and one stops at a
partially-converged loss and guards the sampling path, where the shipped defaults
reproduce the corpus and the old ones miss.

## Best-of-n

Rather than shipping whichever sample came out first, the bot draws
`BABBLE_BEST_OF` candidates (default `4`) and keeps the one **the model itself
scores best** — lowest mean per-byte loss under the same `<bos> text` layout the
trainer optimises, with the shared prefix masked out so the comparison is
between the candidates and not between bytes they both copied.

This directly counteracts the failure above. A derailed candidate scores
terribly under the model that derailed it, so it loses to the two-out-of-five
draws that came out right. It needs no reward model, no second network and no
preference head; at this scale the model is the only scorer that exists.

The candidates are drawn as one batch and scored in one forward pass, so the
cost is sub-linear in `n`. Decode uses a **preallocated KV cache** on CPU: the
prompt is prefilled once, then each new byte is a single-token forward instead
of redoing attention over the whole prefix. That is what keeps a `best_of=4`
reply snappy on two threads at `max_new_tokens=256`. `BABBLE_BEST_OF=1` turns
best-of off.

## CPU-first decode

babble is built for a couple of CPU threads, not a GPU. The user-visible win is
**decode**. The training step is slightly *slower* — about 5% (362 ms → 381 ms
per step at batch 16, block 128, two threads), which is the tanh-GELU tradeoff
below paying off at decode shapes and costing at train shapes. A background
trainer that already rests between cycles can afford that; a Discord reply
cannot afford the decode cost.

- **CPU-only torch** via uv's pytorch-cpu index — no CUDA wheel, no device
  offload in train or inference (`babble/cpu_runtime.py`).
- **KV-cached decode** in `model.py` / `generate.py` so Discord replies do not
  recompute the growing prefix every byte. On `best_of=4` at two threads: 5.5×
  at `max_new_tokens=64`, 9.4× at 128, 15.1× at the shipped default of 256 —
  the longer the reply, the more the cache is worth.
- **oneDNN / MKL-DNN**, denormal flushing, and a capped thread count from
  `be_polite` / `CheckpointGenerator`. Inference defaults to 4 threads
  (`BABBLE_INFER_THREADS`); training stays at 2. Eight threads is a regression
  on this box (see `CPU_INFERENCE.md`).
- **Dynamic int8 Linears** at load, opt-in via `BABBLE_QUANTIZE=1`. On the
  34.1M live checkpoint that is ~1.5× tok/s at +0.002 bits/char — off by
  default so serving quality never regresses without asking for it; see
  `CPU_INFERENCE.md` for the measured tradeoff. `torch.compile` stays off
  too: ~35 s to first forward.
- **tanh-approx GELU**, `Identity` instead of `Dropout(0)`, tied embeddings,
  bias-free projections — fewer ops for the same 3.3M-param shape. The GELU
  form is a decode-shape win and a train-shape loss on this box: at `T=1` it
  beats the erf form (847 ms vs 955 ms on `best_of=4`, 256 tokens), while on
  a full training activation oneDNN's fused erf path wins (7.4 ms vs 11.8 ms
  fwd+bwd). GELU is stateless, so checkpoints still load; the forward is not
  bit-identical to the old erf form, but the drift on live weights is
  negligible — mean per-token loss 13.692948 → 13.693907 on `base.pt`, with
  the greedy reply byte-identical.
- **AdamW `foreach`** on the CPU training path; optional `BABBLE_TORCH_COMPILE=1`
  for a long training run (off by default). Compiled modules are unwrapped
  before every checkpoint write so `state_dict` keys stay plain Babbler keys —
  never `_orig_mod.*`.

Checkpoint weights stay loadable across this change: the KV cache is
runtime-only, and saved key names match `main`.

## What "RL harder" actually means here

ro asked for a better reward loop. Taken literally that means RLHF, and RLHF at
nineteen rows is not a real option: the first thing it needs is a reward model,
and a reward model fit on nineteen rows is noise wearing a network's clothes.
There is nothing here to train one *on*.

So the two things above are the honest version of the same idea, and between
them they cover what a reward loop would have bought:

- **Correction upweighting** *was* the "reward" half — the rows a human actually
  typed counted for `BABBLE_CORRECTION_BOOST`× more than the rows the bot merely
  wasn't told off for. **It is gone**, along with the paired objective it
  weighted. There is no "chosen" answer in an unlabelled corpus to be worth more
  than anything else, and pretending one row of English deserves 3× another
  row's gradient was only ever defensible when one of them was a stated
  correction. The setting survives on the stored rows as metadata and is read by
  nothing.
- **Best-of-n** is the "policy improvement" half, at inference instead of in
  training, and it is untouched. It picks the best of several draws using the
  only scorer that exists at this scale — the model itself — and costs nothing
  but a second of CPU.

**What was deliberately not built:** PPO, GRPO, DPO, or anything else with a
reward model or a preference optimiser in it. Worth saying that the
*corrections* config is already a preference dataset — every row is
`(prompt, rejected, chosen)` — so [DPO](https://arxiv.org/abs/2305.18290) is the
method to reach for if a preference stage is ever wanted: closed-form over the
pairs, no reward network, no rollouts. That is exactly why the pairs are still
captured and still published even though nothing trains on them. It is not
wanted yet: there are fewer than a dozen usable pairs, which makes a "learned
preference" a coincidence with a loss curve attached, and the base model is
still the underfit thing. Pretrain first, align later — in that order, which is
the order the pivot to a corpus put things in.

## Validation

A tiny, fast-learning corpus makes train loss meaningless on its own — the
model can just memorise a dozen rows. So a fraction of the consented rows
(`BABBLE_VAL_FRACTION`, default `0.2`) is held out and never trained on, only
scored, in eval mode, at every checkpoint.

- **Deterministic, not shuffled.** Which side of the split a row lands on is
  decided by a stable hash of its id, not its position in the file or a random
  draw. Same corpus in, same split out, restart after restart.
- **Exactly the configured fraction, not a coin flip per row.** The holdout is
  the lowest-hashed `round(val_fraction × n)` rows. It used to be every row
  whose hash happened to fall under `val_fraction`, which is a *binomial draw*
  — and at this corpus size that draw is wild. The live trainer held out **10
  of 21 rows** at `val_fraction=0.2`, halving a corpus that only had 21 rows in
  it. The hash itself is fine (measured over 10,500 real row ids: flat deciles,
  0.1958 realised against a 0.2 target); the scheme was the problem. Simulating
  4,000 21-row corpora under the old rule, the holdout ranged from 0 to 12. A
  mean of 4.2 is no comfort when any single run is the one you are training.

  The cost is that a couple of rows near the cut can change sides as the corpus
  grows, since `k` grows and a new row may hash below the boundary. That churn
  is bounded and never depends on file position — and it is well worth the
  split actually being the size it claims.
- **Small corpora skip it entirely.** Below `BABBLE_VAL_MIN_ROWS` (default
  `20`) rows, holding out 20% could starve training of most of what little
  data there is, so validation is disabled outright — every row trains, and
  the log line and feed post say plainly that validation is off and why,
  rather than printing a val loss that means nothing.
- **Eval-only, always.** The validation pass runs in `model.eval()` mode with
  no gradient and no optimizer step — it cannot move a weight or an AdamW
  moment. It reuses exactly the rows `!babble consent` and the [content
  filter](#content-filter) already let through training, so a held-out row is
  held to the same rules as a trained-on one.
- **Overfitting is reported, never acted on.** If val loss has risen while
  train loss fell since the last checkpoint, the log line and the feed post
  carry a plain flag saying so. Nothing stops, no hyperparameter changes —
  this is a signal for whoever's watching, not an automatic decision.

- **The probe says where its prefix came from.** Every checkpoint takes the
  opening bytes of one corpus row and prints what the model writes after them.
  There is no expected answer printed alongside, because an unlabelled corpus
  has none. Identical bad output means opposite things depending on where that
  prefix came from: seeded with a hardcoded string the model has never seen,
  nonsense is expected; seeded with a row it has trained on, nonsense is a bug.
  The probe walks the **train** split, and the log line and feed post say which
  case it is explicitly (`probe_side`), so the memorisation question is
  answerable from the feed at every checkpoint instead of being ambiguous.
  The prefix is sliced out of the row's own bytes rather than rebuilt from its
  words, so it is a genuine prefix — rejoining words with single spaces would
  feed the model a string that never appeared in training and then blame the
  model for not recognising it.

`babble train` never trains on held-out rows; only their loss is read.

## Collection feed

While no trainer is running — which is the whole of the collection phase — the
feed channel does not go silent. Instead of training progress it reports
**collection**: the moments the corpus actually changes. It posts to the same
`BABBLE_LOG_WEBHOOK_URL` webhook the training feed uses, so one channel shows
whichever phase babble is in, and it obeys the same two rules — silent when the
webhook is unset, every send failure logged and swallowed so a Discord outage
can never break capture.

```
🌱 corpus +1
> `hey there booper` — a ping · u_9f2c…
now 55 rows · 1,031 chars · 8 contributors
📈 milestone — 50 corpus rows collected
✅ u_9f2c… opted in — their messages now go into the corpus
📡 u_9f2c… opened channel c_1a2b… with `!babble all` — everything they say there is now collected
🗑️ u_9f2c… withdrew — purged 3 stored row(s) (2 corpus · 1 correction)
📤 published 78 row(s) to https://huggingface.co/datasets/kowo-co/babble — corpus grew +10 rows / +240 chars since the last publish
```

- **A row arriving** shows the text collected (run through the [content
  blocklist](#content-filter) and the same `neuter_sample` the training probe
  uses, so it can never ping and a blocked term is withheld), which surface it
  came from — a ping, a reply, a DM, or a widened `!babble all` channel — the
  contributor's **pseudonym only**, and the running totals.
- **Bursts coalesce.** Someone who ran `!babble all` and is typing produces one
  message listing the rows that arrived inside a few-second window, not one post
  per message.
- **Consent changes** are posted too, because they change what may be collected
  at all: a grant, a channel widened or narrowed, an opt-out, a withdrawal (with
  how many rows the purge removed).
- **Milestones** every N rows and N characters, with the interval scaled to the
  corpus size — every 25 rows while it is tiny, every 500 once it is in the
  thousands — so growth stays legible without one post per row and without spam
  at 10,000 rows.
- **Pseudonymous, absolutely.** No Discord id, username, or raw identifier ever
  appears in a feed message; a person is a salted hash like `u_9f2c…`. Channel
  ids in a widen event are the pseudonymous `c_…` hash, not the raw room id.

## Training feed

When `babble train` runs, it posts a short message to the same channel on
start and every checkpoint — step, current loss and its delta from the last
checkpoint, how many corpus rows are being trained on, and the sample
generation, which is the actual point:

```
🚀 cycle 1 starting · 154 stored → 154 training, 0 dropped · 124 train / 30 val rows
   · 131 examples ≈4,208 tokens · batch 8 @ lr 0.001
🔁 cycle 1 · step 650 · loss 2.1840 (-0.312) · 124 rows
   val 2.4120 (+0.018) · 30 held out
> continuation _(trained)_: `hey how` → ` are yout hi ther`
```

**The probe is a continuation, and says so.** It seeds the model with the first
few words of a real row from the *training* split and shows what it writes next.
There is no expected-answer line, because an unlabelled corpus has no expected
answers in it — printing a field called `expected` would be the single most
misleading thing this post could do. `_(trained)_` says the prefix came from a
row the model has actually seen, which is what makes bad output here meaningful:
nonsense continuing a memorised row is a bug, nonsense continuing a string it
has never seen is Tuesday.

If the corpus is too small to hold anything out yet, the val line reads `val:
disabled — <reason>` instead of a number — see [Validation](#validation).

**The trainer and the bot are separate processes** — `babble train` has no
Discord login and never will, that separation is deliberate. So this posts
over a plain Discord **webhook** (one HTTPS POST, no login, no gateway, nothing
to keep alive) rather than merging the two processes or building a file-tailing
relay between them. Set `BABBLE_LOG_WEBHOOK_URL` and it turns on; leave it unset
and nothing changes — no channel configured means no posting, no errors, same
behaviour as today.

- **Best-effort, always.** A bad URL, no network, Discord being down or rate
  limited — every failure is logged and swallowed. Posting never touches
  training; a failed post is exactly as consequential as a dropped log line.
- **One post per checkpoint**, and `BABBLE_LOG_EVERY_N` throttles further (post
  only every Nth checkpoint) so a long run doesn't flood the channel.
- **Never a ping.** The sample is arbitrary model output — escaped, truncated,
  and sent with `allowed_mentions` cleared, with mention markup broken on top
  of that as a second layer. It cannot ping `@everyone`, a role, or a user, no
  matter what the model emits.
- **The same content filter applies here too.** A sample that matches the
  [content blocklist](#content-filter) is withheld from the post rather than
  sent, the same as anywhere else the model's output reaches Discord.
- Everything posted also lands in `logs/babble.jsonl` / `logs/babble.log` as
  usual — the feed is a second, best-effort delivery of the same events, not a
  replacement for the log.

## Watching it

Everything meaningful is logged, twice: `logs/babble.jsonl` for machines and
`logs/babble.log` for humans.

```bash
babble logs -n 40         # recent activity
babble logs --follow      # live tail
babble logs --json        # the structured stream
babble summary            # one-shot state of everything
```

```
2026-08-14T02:56:59+00:00  train.checkpoint   step=200 loss=0.0016 rows=13 sample=hey!
2026-08-14T02:57:00+00:00  capture.correction row=8f2a user=u_5966c0bd weight=1 total_rows=14
2026-08-14T02:57:00+00:00  capture.skipped    signal=correction reason=no_consent missing=signal_author
```

Logged: startup and connect, every ping, every generation with its sampling
params and checkpoint step, every consent prompt/accept/decline/withdraw, every
captured correction and 👍, every reply that was answered but not learned from
for want of the [`>>` marker](#why-corrections-need-the--marker)
(`capture.unmarked`), every skipped-for-no-consent event **with its reason
but never its content**, every training cycle, every checkpoint, every
resume-after-kill, every export, every blocklist rejection **with a content hash
but never the text**, and every Discord feed post that failed to send.

A message the bot decides *not* to answer is never silent either: `bot.dropped`
records the reason (not addressed to it, author is a bot, …) plus the
pseudonymised channel and guild it came from, and a failed reply logs `bot.error`
with `reason=forbidden` (missing permissions) or `reason=http_error` (anything
else) so a permissions gap in one server doesn't read the same as a bug. On
connect, `bot.guild` logs each guild it's in and how many of its text channels it
can actually see and send in — the fastest way to catch "the bot can see the
server but not this channel" without a manual API poke.

Reading a log never mutates it — logs are opened append-only, never truncated on
read or on restart, and rotated by size (`babble.jsonl.1`, `.2`, …). Identifiers
are pseudonymised with the same hash the dataset uses, and message content is
logged only for people who have opted in.

## Publishing to HuggingFace

```bash
babble export                                   # build ./export, push nothing
babble export --push --repo kowo-co/babble-corrections   # needs HF_TOKEN
```

`export` writes **two** files plus a dataset card, containing consented rows
only, with authors as salted hashes:

| file | HF config | what's in it |
| --- | --- | --- |
| `export/data/corpus.jsonl` | `default` | the unlabelled corpus — `{id, text, author, source, created_at}` |
| `export/data/train.jsonl` | `corrections` | the `(prompt, rejected, chosen)` pairs, same path they always had |

The corpus is the `default` config because it is the content: it is what the
model trains on and what anyone loading the dataset most likely wants. The pairs
keep their original path so an existing download script pointed at the file
keeps working.

Each config is checked against the grant that actually covers it — the corpus
against `corpus`, the pairs against `corrections` — so somebody who opted in
under the old notice and has not answered the new one appears in the second file
and not the first. Rows are content-addressed and stably sorted, so re-running
produces byte-identical output and re-pushing is a no-op.

The guild and channel a corpus row came from are stored locally but **not**
published. They are pseudonymous, so publishing them would identify nobody; they
are simply not needed by anyone downloading a text corpus.

Three guards: the export **aborts** if an author field is not a pseudonym (that
would mean a bug), and it **drops** any row whose text contains a known raw
Discord id or mention markup, or that matches the [content blocklist](#content-filter),
rather than publishing it. All three apply to both files.

**The dataset auto-publishes on corpus growth.** In the collection phase there
is no trainer and so no checkpoints, so publishing is keyed to the **data**: the
bot pushes once the corpus has grown by `BABBLE_HF_PUBLISH_EVERY_ROWS` rows
(default **10**) **or** `BABBLE_HF_PUBLISH_EVERY_CHARS` characters (default
**2000**) since the last publish, whichever comes first. It builds the export
and pushes it to `BABBLE_HF_REPO` — through the exact same consent and blocklist
guards as a manual `--push`, with `HF_TOKEN` read from the environment the same
way. Set both to `0` to turn it off and publish by hand only. See
`babble/publish.py`.

- **Keyed to a persisted baseline.** The corpus size at the last publish is
  written to `data/publish_state.json`, so a bot restart neither loses the
  baseline nor re-publishes on boot, and a growth threshold — not a wall-clock
  timer — is what triggers a push.
- **Skips a no-op push.** If the export is byte-identical to the last one
  actually pushed (same content hash), nothing is sent — e.g. when every new row
  was unconsented or blocklisted and dropped at export.
- **Never breaks collection.** A failed push (bad token, no network, HF down,
  rate limited) is logged and reported once in the [collection feed](#collection-feed);
  capture keeps going. An attempt advances the baseline whatever its outcome, so
  a failure does not retry on every subsequent message — it waits for another
  threshold of growth, and the next full export carries everything anyway.
- **Reported in the feed** as a one-line `📤 published N row(s) to <url>`, or
  `⚠️ dataset publish failed — <error>` on failure.

**The trainer still auto-publishes on checkpoints too**, when one is run by hand:
every `BABBLE_HF_PUBLISH_EVERY` checkpoints written (default **20**), through the
same guards. That path only fires while a trainer is running, which is why the
growth-keyed publish above exists — it is what keeps the public dataset live
during a collection phase with no trainer.

## Configuration

Copy `.env.example` to `.env`. The only thing the bot strictly needs is
`BABBLE_DISCORD_TOKEN` (create the app at
<https://discord.com/developers/applications> and enable the **Message Content**
privileged intent). Everything else has a working default, including the
[training feed](#training-feed) (`BABBLE_LOG_WEBHOOK_URL`) and the
[content blocklist](#content-filter) (`BABBLE_BLOCKLIST_PATH`), both of which
are entirely optional.

The knobs that decide what the bot sounds like:

| variable | default | what it does |
| --- | --- | --- |
| `BABBLE_TEMPERATURE` | `0.5` | sampling temperature — [`1.0` was the babble](#why-it-babbled-at-loss-002) |
| `BABBLE_TOP_K` | `40` | truncate sampling to the top k tokens |
| `BABBLE_TOP_P` | `0.9` | nucleus sampling; `1.0` disables it (top-k still applies) |
| `BABBLE_REPETITION_PENALTY` | `1.15` | HF-style penalty on prompt + generated tokens; `1.0` disables it -- **flat**: a token seen 80 times is discounted the same as one seen once |
| `BABBLE_FREQUENCY_PENALTY` | `0.12` | additive penalty that grows with each repeat (`logit -= frequency_penalty * count`) -- the fix for the flat penalty above letting an induced loop run away; largest value in testing that broke every induced loop without visibly degrading normal replies; `0.0` disables it |
| `BABBLE_PRESENCE_PENALTY` | `0.0` | flat, one-time additive penalty on anything already seen at all, count aside; off by default -- testing found no loop it stopped that `frequency_penalty` did not already stop |
| `BABBLE_NO_REPEAT_NGRAM_SIZE` | `0` | hard-ban whichever next token would complete an already-seen n-gram of this size; off by default -- on top of `frequency_penalty=0.12` it only ever fired on ordinary phrase reuse and fragmented otherwise-coherent replies; a last-resort circuit breaker to enable if a future prompt gets past the frequency penalty alone |
| `BABBLE_MAX_NEW_TOKENS` | `256` | longest reply, in bytes |
| `BABBLE_BLOCK_SIZE` | `512` | context window in bytes; changing it [invalidates checkpoints](#pretraining) (harmless -- every run starts from random init) |
| `BABBLE_TRAIN_TRIGGER_ROWS` | `100` | new corpus rows that [re-fire training](#pretraining); `0` = manual only |
| `BABBLE_TRAIN_STEPS` | `1600` | training step [ceiling](#pretraining), not a target -- the best-val checkpoint may win earlier; also the cosine schedule length |
| `BABBLE_TRAIN_MIN_STEPS` | `600` | floor before the [early stop](#pretraining) may fire; sits inside the step ceiling by design |
| `BABBLE_TRAIN_STALL_MARGIN` | `0.05` | val must exceed the best by more than this ([the measured noise band](#pretraining)) to burn patience; `0` restores the old hair-trigger |
| `BABBLE_TRAIN_PATIENCE` | `3` | stop training after this many non-improving [checkpoint intervals](#pretraining); `0` = never |
| `BABBLE_LEARNING_RATE` | `3e-4` | pretrain AdamW learning rate (sweep winner; the old `1e-3` lost by ~0.15 nats of held-out val across three seeds) |
| `BABBLE_TRAIN_COSINE` | on | cosine-anneal the LR to a tenth of itself over the step budget; set `0` for a constant LR |
| `BABBLE_DROPOUT` | `0.2` | dropout used during training (eval/generation always run with it off) |
| `BABBLE_WEIGHT_DECAY` | `0.01` | AdamW weight decay |
| `BABBLE_TRAIN_SYNTHETIC` | on | mix [labelled synthetic corpus rows](#synthetic-corpus-rows) into the train side (val stays 100% real); set `0` to train on human rows only |
| `BABBLE_POST_TRIGGER_PAIRS` | `10` | new correction pairs that [re-fire post-train](#post-training-on-the-correction-pairs); `0` = manual only |
| `BABBLE_POST_STEPS` | `200` | post-train step [ceiling](#post-training-on-the-correction-pairs), not a target -- the best-val checkpoint may win earlier |
| `BABBLE_POST_PATIENCE` | `3` | stop post-train after this many non-improving [checkpoint intervals](#post-training-on-the-correction-pairs); `0` = never |
| `BABBLE_POST_LEARNING_RATE` | `1e-4` | post-train LR — [no longer borrows the pretrain LR](#post-training-on-the-correction-pairs) |
| `BABBLE_POST_REHEARSAL` | `0.5` | fraction of each post-train batch that is plain corpus text under the pretrain objective |
| `BABBLE_POST_MIN_PAIRS` | `100` | below this many trainable pairs post-train refuses to run (`--force` overrides) |
| `BABBLE_POST_GATE_MARGIN` | `0.05` | [promotion gate](#post-training-on-the-correction-pairs): a candidate worse than the pretrain snapshot by more than this on held-out corpus val is not written to `latest.pt`; negative disables |
| `BABBLE_POST_LAYOUT` | `continuation` | what post-train teaches: `continuation` (the layout serving actually uses) or `pair` (the historical `<bos> prompt <sep> response` layout) |
| `BABBLE_SERVE_LAYOUT` | `continuation` | what decode layout `CheckpointGenerator` actually serves with: `continuation` (`generate.best_continuation`) or `pair` (`generate.best_of`) -- see below |
| `BABBLE_BEST_OF` | `4` | [candidates drawn per reply](#best-of-n); `1` turns it off |
| `BABBLE_TRAIN_THREADS` | `2` | CPU threads for **training** |
| `BABBLE_INFER_THREADS` | `4` | CPU threads for decode (8 is slower; see `CPU_INFERENCE.md`) |
| `BABBLE_QUANTIZE` | off | `1` turns on dynamic int8 on Linear layers at load (faster, small bits/char cost) |
| `BABBLE_TORCH_COMPILE` | off | set `1` to `torch.compile` the model (slow first forward) |
| `BABBLE_VAL_FRACTION` | `0.2` | [share of corpus rows held out](#validation) |
| `BABBLE_VAL_MIN_ROWS` | `20` | corpus size below which validation is skipped |

`BABBLE_SERVE_LAYOUT` does not auto-detect: the checkpoint file itself carries
no flag saying which layout it was trained on, so a checkpoint SFT'd on
prompt/response pairs (e.g. SSH's booper-chat, SFT'd on
`mookiezi/Discord-Dialogues`) must be served with `BABBLE_SERVE_LAYOUT=pair`,
or it keeps writing more of the user's message instead of answering it. **The
live install currently sets `BABBLE_SERVE_LAYOUT=pair`** for exactly that
checkpoint. An ordinary pretrained/`continuation`-post-trained checkpoint
needs no override -- the default already matches what it was trained on.

`correction_weight` and `approval_weight` still exist in `config.py` and are
still written into the `weight` field of stored correction rows, but **nothing
reads that field any more** — see [the pivot](#the-pivot-from-pairs-to-a-corpus).
`BABBLE_CORRECTION_BOOST` is read by nothing at all: it multiplied the stored
weight at training time, and that training path is gone.

There is no environment variable for whole-channel collection, on purpose. It is
[a consent decision made per person, in the channel, with a command](#widening-it-one-person-and-one-channel-at-a-time),
not a deployment setting somebody can flip on for a whole server at once.

The [correction marker](#why-corrections-need-the--marker) is `CORRECTION_MARKER`
in `babble/config.py` rather than an environment variable — it is part of the
protocol people are taught in the footer, not a per-deployment tuning knob.

## Running it for real

The **bot** is the only standing service. It collects the corpus, posts the
[collection feed](#collection-feed), and publishes the dataset on growth — all
with no trainer involved. Run it as a `systemd --user` unit so it survives
logout:

```ini
# ~/.config/systemd/user/babble-bot.service
[Unit]
Description=babble discord bot
[Service]
WorkingDirectory=%h/babble
EnvironmentFile=%h/babble/.env
ExecStart=%h/babble/.venv/bin/babble bot
Restart=always
RestartSec=10
[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now babble-bot
babble logs --follow
```

**There is no separate trainer service.** The running bot itself is what fires
training: after every fresh corpus row it checks the [+N-row
trigger](#pretraining) and, when due, launches `babble train` as a detached,
low-priority subprocess — no unit to install, nothing else to enable. Training
is never a standing, continuously-cycling process (`babble train --loop` was
retired for exactly that reason). To train on demand from a terminal:

```bash
babble train --force
```

### Running a second bot side by side

Nothing in babble is a singleton: every path and knob comes from the
environment, the model's shape and vocab come from the checkpoint file itself,
and bots ignore each other's messages (bot-authored messages are dropped at
capture). So a second bot serving a *different model under a different Discord
account* — for A/B-ing two checkpoints in the same channel — is one more
systemd unit pointed at its own env file, sharing the installed code and venv:

```bash
# Own state root -- data (consent, salt), checkpoints, logs all separate.
mkdir -p ~/babble-boopit
cp deploy/env.boopit.example ~/babble-boopit/.env   # then edit: token + paths

# The checkpoint pair, from its HF repo:
deploy/fetch-hf-checkpoint.sh ProCreations/boopit-1 ~/babble-boopit/checkpoints

cp deploy/babble-boopit.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now babble-boopit
BABBLE_LOG_DIR=~/babble-boopit/logs babble logs --follow   # wait for bot.ready
```

Three things the env file must get right (see `deploy/env.boopit.example`,
which sets all three):

- **One token, one process.** The second bot needs its own Discord
  application and token; two processes on one token bounce each other off
  the gateway.
- **`BABBLE_SERVE_LAYOUT` must match the served checkpoint** — same
  invariant as the main bot, nothing auto-detects it.
- **Zero the training triggers** (`BABBLE_TRAIN_TRIGGER_ROWS=0`,
  `BABBLE_POST_TRIGGER_PAIRS=0`) so corpus growth can never overwrite the
  promoted checkpoint with a from-scratch model, and zero the publish
  cadences — one comparison bot must not double-publish the dataset.

## Keeping the live install current

The bot runs from a **plain clone**, and drift is invisible if nothing checks
for it: a checkout can sit behind `main` — or worse, have its `origin` pointed
somewhere that will never pull anything real again — while everything still
*looks* healthy. `deploy/update-live.sh` is a small, idempotent script built to
run on a timer and catch exactly that:

**Never hand-edit tracked files inside `~/babble-live` directly.** It happened
once (2026-08-22): a run patched `generate.py` and friends straight into the
live checkout instead of landing the change on `main`, which left `git
status` dirty and silently blocked `babble-update.timer`'s own drift check
(`"working tree has uncommitted tracked changes -- refusing to merge"`,
logged every five minutes) — so two full merged PRs' worth of fixes sat on
`main` while the channel was told they were live. Land the change on `main`
like any other change, then either wait for the timer or run `systemctl
--user start babble-update.service` once to pull it in immediately. If you
truly need to poke the live checkout by hand for a one-off diagnosis, `git
checkout -- <file>` (or `git reset --hard origin/main`) before you leave it,
so the next scheduled check finds a clean tree and can keep doing its job.

- **Fails loudly**, never silently "fixes" anything, if `origin` isn't actually
  `kowo-co/babble` — a wrong remote is the kind of bug that hides for weeks, so
  it gets a non-zero exit and a log line, not an auto-repoint.
- Fetches `origin/main`; if the checkout is already current it exits `0` and
  touches nothing — no restart, one log line.
- If behind, refuses to merge over uncommitted tracked changes, then
  `git merge --ff-only` only — never a real merge, never a rebase, never
  force.
- Syncs dependencies (`uv sync`) only when `uv.lock` actually changed.
- **Skips the restart** (exit `0`, try again next tick) if a training run is
  currently in flight, so an update can never kill it mid-write.
- Restarts `babble-bot` and then *proves* it came back — polls the log for a
  fresh `bot.ready` within a timeout — rather than trusting that
  `systemctl restart` returning success means the bot is actually serving.
- Logs every action, and every no-op decision, to the same `logs/babble.log` /
  `logs/babble.jsonl` the bot writes, and records the outcome of the last
  check in `data/update_state.json`.
- **A refusal or failure is never just a failed systemd unit.** The
  2026-08-22 incident above repeated identically every five minutes for two
  days with nothing surfacing it outside a log line nobody was tailing --
  `data/update_state.json` now also tracks `consecutive_failures` and
  `commits_behind`, and on the first failure, then every
  `BABBLE_UPDATE_ALERT_EVERY_N`-th one after that (default `5`), the script
  drops a `data/UPDATE_FAILING` marker and, if `BABBLE_LOG_WEBHOOK_URL` (the
  same webhook the [training feed](#training-feed) uses) is set, posts one
  line to it. Both clear automatically on the next clean run. Unconfigured
  webhook is silent, same convention as the rest of the repo.
- `deploy/update-live.sh --check` prints the same state in one greppable,
  network-free line -- `status=clean|dirty behind=<n> ahead=<n> last_action=<...>
  consecutive_failures=<n> checked_at=<...>` -- and exits `0` only when the
  tree is clean *and* HEAD is not ahead of origin (divergence is as loud as
  dirtiness), instead of requiring a `git status` on the box.
- A checkout that cannot fast-forward -- HEAD has commits origin does not --
  is refused with `last_action=skipped_diverged` on the same first-then-every-
  Nth alert path as `skipped_dirty`. The 2026-08-25 live box had five local-only
  commits and the updater failed every five minutes with
  `Not possible to fast-forward` while looking no louder than a dirty tree.

### Installing the timer

The service and timer units, and the script they run, live in `deploy/` and
ship with the repo — installing them is symlinking or copying two unit files
and enabling one:

```bash
mkdir -p ~/.config/systemd/user
ln -sf ~/babble-live/deploy/babble-update.service ~/.config/systemd/user/
ln -sf ~/babble-live/deploy/babble-update.timer   ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now babble-update.timer
```

That's it — no separate install step for the script itself. It lives inside
the live checkout, so it updates *itself* the same way it updates everything
else, and the next scheduled run always uses whatever version of the script
just landed on `main`.

The units assume the live checkout is `~/babble-live` and the bot's unit is
named `babble-bot`, matching [the bot service above](#running-it-for-real).
Both — and every other path the script touches — are overridable by
environment variable rather than hardcoded, so the same script runs anywhere:

| variable | default | |
| --- | --- | --- |
| `BABBLE_LIVE_DIR` | `~/babble-live` | the checkout to keep current |
| `BABBLE_UPDATE_REMOTE` | `https://github.com/kowo-co/babble.git` | the `origin` that must be configured |
| `BABBLE_UPDATE_BRANCH` | `main` | branch to track |
| `BABBLE_BOT_UNIT` | `babble-bot` | the systemd `--user` unit to restart |
| `BABBLE_UPDATE_RESTART_TIMEOUT` | `90` | seconds to wait for `bot.ready` after a restart |
| `BABBLE_UPDATE_ALERT_EVERY_N` | `5` | alert on the 1st consecutive failure, then every Nth after that |
| `BABBLE_TRAIN_SUBCOMMANDS` | `train post-train` | space-separated `babble` subcommands that count as "training in flight" |

The timer runs on a wall-clock schedule (`OnCalendar=*:0/5`, every 5 minutes)
with `Persistent=true`, so a tick missed while the box was off still fires on
the next start, and a fresh boot gets an early check via `OnBootSec=2min`.
A wall-clock schedule is what makes it keep firing forever regardless of when
it was enabled — `OnUnitActiveSec=`, which the timer used to rely on alone,
re-arms relative to the timer unit's own last activation, so without a
recurring `OnCalendar=` it goes "elapsed" after the initial boot-relative shot
and never fires again. To change the check interval, edit `OnCalendar=` in
`deploy/babble-update.timer` (a few minutes is a sane default — the service
is a fast no-op whenever `main` hasn't moved) and `systemctl --user
daemon-reload && systemctl --user restart babble-update.timer`.

### Checking whether it's actually current

Four ways to answer "is booper running the latest code?" without shelling
into the box:

```bash
babble summary                              # last line: running commit vs origin/main
babble logs -n 20                           # recent update.* events, interleaved with everything else
cat ~/babble-live/data/update_state.json    # exactly what the last check decided, and when
deploy/update-live.sh --check               # one greppable line: clean/dirty, behind, ahead, last action
```

`babble summary`'s `code` line reports the commit actually running (read
fresh, locally) against `origin/main` **as of the last scheduled check** — it
never fetches over the network itself, so asking is always instant:

```
code 6af568a · current with origin/main (checked 2026-08-17T07:15:23+00:00)
code 6af568a · BEHIND origin/main (a2250af...) as of 2026-08-17T07:20:00+00:00
code 6af568a · origin/main: unknown (self-update timer has never checked)
```

`systemctl --user status babble-update.timer` shows when it last fired and
next will; `systemctl --user start babble-update.service` runs one check
immediately instead of waiting for the timer.

## Layout

```
babble/
  tokenizer.py   byte-level vocab: 256 bytes + <pad> <bos> <sep> <eos>
  model.py       the transformer — random init, KV cache, CPU-friendly ops
  cpu_runtime.py force CPU / oneDNN / thread caps; optional torch.compile
  generate.py    sampling (continuations and pairs), hot-reloading checkpoints
  trainer.py     the polite trainer: random init, human corpus, best-val + trigger
  post_state.py  the post-train +N-pair trigger and pair filtering (torch-free)
  posttrain.py   post-train: fine-tune the pretrained checkpoint on the pairs
  synthetic.py   postulated-prompt pairs from reply-shaped corpus rows (torch-free)
  core.py        ALL bot behaviour, with zero Discord imports
  bot.py         thin discord.py adapter — the only file that imports discord
  consent.py     who agreed and to what; two scopes; fails closed
  blocklist.py   content filter -- normalisation + word-boundary matching
  blocklist.txt  the (small, starter) list itself
  corpus.py      the unlabelled corpus — what the model trains on
  store.py       the correction triples, pseudonymous on disk
  backfill.py    the idempotent pairs -> corpus migration, run every cycle
  exchanges.py   what it said and to whom, so corrections find their target
  export_hf.py   dataset + card, consent and blocklist re-checked, ids guarded
  publish.py     growth-keyed auto-publish: push when the corpus grows enough
  discord_feed.py collection + training feeds -> a Discord webhook, best-effort
  logs.py        append-only structured + prose event log
  stats.py       snapshot and loss curve rendering
  cli.py         `babble <command>`
```

`core.py` is deliberately Discord-free: it takes plain dataclasses and returns
`Reply` objects. That is why the consent gate, corpus capture, per-channel
widening, correction capture, 👍 handling and purge are all tested against a fake
gateway, with no token and no network.

## Licence

MIT.
