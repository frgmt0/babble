# babble

A language model that starts as **pure noise** and learns to talk only from
people correcting it in Discord.

> "what if we make a model that learns how to talk by people interacting with it
> on discord. So you ping it, then you get a replied response. If you reply to
> that you can correct it with the response you wanted whether it be a gif and
> whatnot."
>
> "No but the from scratch is the most fun. Because it'll be completely random
> and we train on checkpoints in a consistent ultra efficient loop that doesn't
> make the computer unusable"

There is no pretrained base. No corpus. No scraped chat history. `babble` is
~3.3M randomly initialised parameters, and the only data it will ever see is
what people deliberately teach it, one correction at a time.

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

babble fake-data                # a dozen made-up corrections to chew on
babble train --steps 200        # watch the loss drop and the samples change
babble sample --prompt hello    # generate from the newest checkpoint
babble curve                    # the loss curve, as a picture
babble summary                  # step, loss, checkpoints, consent, row counts
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
   produce.
2. **Every reply carries a footer** telling you how to grade it.
3. **You react 👍** — a weak "that was fine", stored with a low weight.
4. **Or you reply starting with `>>`** — the strong signal, and the one that
   actually teaches it:

   ```
   >> hey, what's up
   ```

   Text, an emoji, a URL, a gif link, an uploaded image: whatever follows the
   marker is stored raw.

### Why corrections need the `>>` marker

Without it, *every* reply to one of the bot's messages was stored as the answer
it should have given — so "lol", "wrong" and "what even is that" all went into
the corpus as lessons. That is a corpus full of things nobody meant to teach it,
and it is being trained on and published.

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
explains that your messages, its answers, and your corrections are stored, used
to train it, and **published to a public HuggingFace dataset**. You then choose.

```
!babble accept     opt in
!babble decline    no thanks — it still replies, it just keeps nothing
!babble forget     opt out and delete everything of yours it has kept
!babble consent    show the notice and your current setting
!babble status     how training is going
```

The rules, all enforced in code and covered by tests:

- **Fail closed.** No consent record means no data. Silence is not consent, a
  missing file is not consent, a corrupt file is not consent.
- **Checked at capture time**, not just at reply time — and again at training
  time, and again at export time.
- **Everyone in a row must have opted in.** If Alice asks and Bob corrects, the
  row needs both, because Alice's message gets published too.
- **Withdrawal is retroactive.** `!babble forget` deletes your stored rows, drops
  your pending exchanges, and takes you out of every future export.
- **Raw Discord ids never reach the dataset.** They exist only in two local
  operational files (`data/consent.json`, `data/exchanges.json`). The corpus, the
  logs and the export contain only salted hashes like `u_9f2c4a…`. Usernames are
  never read at all.

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
- new corrections are picked up at the start of each cycle
- the bot hot-reloads `latest.pt` as it appears — no restart needed

**It is safe to kill at any moment.** Checkpoints are written to a temp file and
renamed, so `kill -9` mid-write leaves the previous one intact. `SIGINT`/`SIGTERM`
finish the current step, checkpoint, and exit cleanly. Restarting resumes the
step count, the weights and the AdamW moments from `checkpoints/latest.pt`.

Every checkpoint appends `{step, loss, sample}` to `checkpoints/loss.jsonl` and
prints it, alongside the worst single row and the held-out validation loss:

```
step     100 | loss   0.0263 | worst  1.012 | val   0.0891 | 'hello' -> 'hey!'
```

Only the response tokens are trained on — the prompt is context, not a target.
Corrections carry weight `1.0`, 👍 rows `0.25`, and corrections are then
multiplied again by `BABBLE_CORRECTION_BOOST` (default `3.0`) so a reply someone
actually typed outweighs a row the bot merely wasn't told off for.

The `rejected` field is stored and published but not trained on — see
[what "RL harder" actually means here](#what-rl-harder-actually-means-here).

## Why it babbled at loss 0.02

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
`train.checkpoint` logs `worst_row_loss` and `worst_row_prompt`. That is the
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
scores best** — lowest mean per-byte loss under the same
`<bos> prompt <sep> response <eos>` layout the trainer optimises.

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

- **Correction upweighting** is the "reward" half. The rows a human actually
  typed count for `BABBLE_CORRECTION_BOOST`× more than the rows the bot merely
  wasn't told off for, so a fresh correction moves behaviour within a checkpoint
  or two instead of being averaged into nineteenths.
- **Best-of-n** is the "policy improvement" half, at inference instead of in
  training. It picks the best of several draws using the only scorer that
  exists at this scale — the model itself — and costs nothing but a second of
  CPU.

**What was deliberately not built:** PPO, GRPO, DPO, or anything else with a
reward model or a preference optimiser in it. Worth saying that the corpus *is*
already a preference dataset — every correction row is
`(prompt, rejected, chosen)`, and supervised training only ever looks at
`chosen` — so [DPO](https://arxiv.org/abs/2305.18290) is the method to reach for
if a preference stage is ever wanted: closed-form over the pairs, no reward
network, no rollouts. It is not wanted yet. The corpus has fewer than a dozen
usable pairs, which makes a "learned preference" a coincidence with a loss curve
attached, and the supervised signal is still the underfit thing. Fix the thing
that is underfit first.

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

- **The probe says which side it came from.** Every checkpoint samples one row
  from the dataset and prints it next to that row's expected answer. Identical
  bad output means opposite things depending on where that row sits: on a
  held-out row the model was never trained on that prompt and garbage is
  expected; on a trained row it is a bug. The probe walks the **train** split,
  and the log line and feed post now say so explicitly (`probe_side`), so the
  memorisation question is answerable from the feed at every checkpoint instead
  of being ambiguous.

`babble train` never trains on held-out rows; only their loss is read.

## Training feed

`babble train --loop` posts a short message to a Discord channel on trainer
start, resume-after-kill, going idle, and every checkpoint — cycle, step,
current loss and its delta from the last checkpoint, how many trainable rows
are in the corpus, and the sample generation, which is the actual point:

```
🔁 cycle 3 · step 650 · loss 2.1840 (-0.312) · 128 rows
   val 2.4120 (+0.018) · 26 held out
> 'hello' → `heoll wrold hi`
```

If the corpus is too small to hold anything out yet, that line reads `val:
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

`export` writes `export/data/train.jsonl` plus a dataset card, containing
consented rows only, with authors as salted hashes. Rows are content-addressed
and stably sorted, so re-running produces byte-identical output and
re-pushing is a no-op.

Three guards: the export **aborts** if an author field is not a pseudonym (that
would mean a bug), and it **drops** any row whose text contains a known raw
Discord id or mention markup, or that matches the [content blocklist](#content-filter),
rather than publishing it.

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
| `BABBLE_CORRECTION_BOOST` | `3.0` | how much more a correction counts than a 👍; `1.0` for no boost |
| `BABBLE_VAL_FRACTION` | `0.2` | [share of rows held out](#validation) |
| `BABBLE_VAL_MIN_ROWS` | `20` | corpus size below which validation is skipped |

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
  generate.py    sampling, and hot-reloading the newest checkpoint
  trainer.py     the polite, resumable, duty-cycled training loop
  core.py        ALL bot behaviour, with zero Discord imports
  bot.py         thin discord.py adapter — the only file that imports discord
  consent.py     who agreed; fails closed
  blocklist.py   content filter -- normalisation + word-boundary matching
  blocklist.txt  the (small, starter) list itself
  store.py       the corpus of triples, pseudonymous on disk
  exchanges.py   what it said and to whom, so corrections find their target
  export_hf.py   dataset + card, consent and blocklist re-checked, ids guarded
  discord_feed.py training progress -> a Discord webhook, best-effort
  logs.py        append-only structured + prose event log
  stats.py       snapshot and loss curve rendering
  cli.py         `babble <command>`
```

`core.py` is deliberately Discord-free: it takes plain dataclasses and returns
`Reply` objects. That is why the consent gate, correction capture, 👍 handling
and purge are all tested against a fake gateway, with no token and no network.

## Licence

MIT.
