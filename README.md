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
babble train --steps 200        # watch the loss drop and the samples change
babble sample --prompt hello    # continue a prefix from the newest checkpoint
babble curve                    # the loss curve, as a picture
babble summary                  # step, loss, checkpoints, consent, row counts
babble backfill-corpus          # flatten old correction pairs into the corpus
babble export                   # build the HuggingFace dataset directory
pytest                          # 100+ tests, none of which need a token
```

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
Hub. `babble backfill-corpus` flattens the existing ones into the corpus so the
text people already gave is not stranded in a format nothing reads any more.

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

`babble backfill-corpus` is gated on the `corrections` grant, not the corpus one,
and deliberately so: that text was collected under the corrections notice and has
already been published under it, so re-filing it is the same words in a second
index rather than a new collection. Anything collected *from now on* needs the
corpus grant. Someone who has explicitly declined the corpus scope is skipped by
the backfill anyway.

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

## The trainer

Training happens in the background, off checkpoints, never in the request path.

```bash
babble train --steps 200          # one cycle
babble train --loop               # work, rest, repeat, forever
```

It is built to leave the machine usable:

- `nice 19` and a capped thread count (`BABBLE_TRAIN_THREADS`, default 2)
- duty-cycled: `BABBLE_STEPS_PER_CYCLE` steps, then `BABBLE_REST_SECONDS` idle
- new corpus rows are picked up at the start of each cycle
- the bot hot-reloads `latest.pt` as it appears — no restart needed

**It is safe to kill at any moment.** Checkpoints are written to a temp file and
renamed, so `kill -9` mid-write leaves the previous one intact. `SIGINT`/`SIGTERM`
finish the current step, checkpoint, and exit cleanly. Restarting resumes the
step count, the weights and the AdamW moments from `checkpoints/latest.pt`.

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
cost is sub-linear in `n`. Measured on the shipped 3.3M model at two threads,
four 96-byte candidates take about **1.5s** against 0.6s for one — and that is
the worst case, measured on a *randomly initialised* model, which almost never
emits `<eos>` and so always runs the full `max_new_tokens`. A trained model
stops early and is faster. `BABBLE_BEST_OF=1` turns it off.

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

## Training feed

`babble train --loop` posts a short message to a Discord channel on trainer
start, resume-after-kill, going idle, and every checkpoint — cycle, step,
current loss and its delta from the last checkpoint, how many corpus rows are
being trained on, and the sample generation, which is the actual point:

```
🚀 cycle 3 starting · 154 stored → 154 training, 0 dropped · 124 train / 30 val rows
   · 131 examples ≈4,208 tokens · batch 8 @ lr 0.001
🔁 cycle 3 · step 650 · loss 2.1840 (-0.312) · 124 rows
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

**The trainer and the bot are separate processes** — `babble train --loop` has
no Discord login and never will, that separation is deliberate. So this posts
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

**The trainer also auto-publishes.** Every `BABBLE_HF_PUBLISH_EVERY` checkpoints
written (default **20**, counted by checkpoints, not steps), it builds the
export and pushes it to `BABBLE_HF_REPO` -- through the exact same consent and
blocklist guards as a manual `--push`, with `HF_TOKEN` read from the
environment the same way. Set `BABBLE_HF_PUBLISH_EVERY=0` to turn it off and go
back to publishing by hand only.

- **Skips a no-op push.** If the export is byte-identical to the last one
  actually pushed (same row count, same content hash), nothing is sent.
- **Never breaks training.** A failed push (bad token, no network, HF down,
  rate limited) is logged and reported once in the [training feed](#training-feed);
  training keeps going and the next scheduled publish just tries again -- no
  retry storm, one attempt per scheduled publish.
- **Reported in the feed** as a one-line `📤 auto-published N row(s) to <url>`,
  or `⚠️ auto-publish ... failed -- <error>` on failure.

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
| `BABBLE_TOP_K` | `40` | truncate sampling to the top k bytes |
| `BABBLE_MAX_NEW_TOKENS` | `96` | longest reply, in bytes |
| `BABBLE_BEST_OF` | `4` | [candidates drawn per reply](#best-of-n); `1` turns it off |
| `BABBLE_VAL_FRACTION` | `0.2` | [share of corpus rows held out](#validation) |
| `BABBLE_VAL_MIN_ROWS` | `20` | corpus size below which validation is skipped |

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

Two `systemd --user` units, so both survive logout:

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

```ini
# ~/.config/systemd/user/babble-trainer.service
[Unit]
Description=babble trainer
[Service]
WorkingDirectory=%h/babble
EnvironmentFile=%h/babble/.env
ExecStart=%h/babble/.venv/bin/babble train --loop
Nice=19
CPUWeight=20
Restart=always
RestartSec=30
[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now babble-bot babble-trainer
babble logs --follow
```

## Layout

```
babble/
  tokenizer.py   byte-level vocab: 256 bytes + <pad> <bos> <sep> <eos>
  model.py       the transformer — random init, no pretrained anything
  generate.py    sampling (continuations and pairs), hot-reloading checkpoints
  trainer.py     the polite, resumable, duty-cycled training loop
  core.py        ALL bot behaviour, with zero Discord imports
  bot.py         thin discord.py adapter — the only file that imports discord
  consent.py     who agreed and to what; two scopes; fails closed
  blocklist.py   content filter -- normalisation + word-boundary matching
  blocklist.txt  the (small, starter) list itself
  corpus.py      the unlabelled corpus — what the model trains on
  store.py       the correction triples, pseudonymous on disk
  backfill.py    the one-shot, idempotent pairs -> corpus migration
  exchanges.py   what it said and to whom, so corrections find their target
  export_hf.py   dataset + card, consent and blocklist re-checked, ids guarded
  discord_feed.py training progress -> a Discord webhook, best-effort
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
