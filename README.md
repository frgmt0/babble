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
2. **Every reply carries a footer:**
   `-# like this response? react with 👍 — if not, correct me with a reply.`
3. **You react 👍** — a weak "that was fine", stored with a low weight.
4. **Or you reply with what it should have said** — the strong signal, and the
   one that actually teaches it. Text, an emoji, a URL, a gif link, an uploaded
   image: whatever you send is stored raw.

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

<<<<<<< ours
=======
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

>>>>>>> theirs
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
prints it:

```
step     100 | loss   0.0263 | 'hello' -> 'hey!'
```

Only the response tokens are trained on — the prompt is context, not a target.
Corrections carry weight `1.0`, 👍 rows `0.25`. The `rejected` field is stored
for the dataset but the current loop does not train against it; it is plain
weighted next-byte prediction on accepted responses.

<<<<<<< ours
=======
## Training feed

`babble train --loop` posts a short message to a Discord channel on trainer
start, resume-after-kill, going idle, and every checkpoint — cycle, step,
current loss and its delta from the last checkpoint, how many trainable rows
are in the corpus, and the sample generation, which is the actual point:

```
🔁 cycle 3 · step 650 · loss 2.1840 (-0.312) · 128 rows
> 'hello' → `heoll wrold hi`
```

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

>>>>>>> theirs
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
captured correction and 👍, every skipped-for-no-consent event **with its reason
but never its content**, every training cycle, every checkpoint, every
<<<<<<< ours
resume-after-kill, and every export.
=======
resume-after-kill, every export, every blocklist rejection **with a content hash
but never the text**, and every Discord feed post that failed to send.
>>>>>>> theirs

Reading a log never mutates it — logs are opened append-only, never truncated on
read or on restart, and rotated by size (`babble.jsonl.1`, `.2`, …). Identifiers
are pseudonymised with the same hash the dataset uses, and message content is
logged only for people who have opted in.

## Publishing to HuggingFace

```bash
babble export                                   # build ./export, push nothing
babble export --push --repo kowo-co/babble-corrections   # needs HF_TOKEN
```

Nothing is ever pushed automatically. `export` writes `export/data/train.jsonl`
plus a dataset card, containing consented rows only, with authors as salted
hashes. Rows are content-addressed and stably sorted, so re-running produces
byte-identical output and re-pushing is a no-op.

<<<<<<< ours
Two guards: the export **aborts** if an author field is not a pseudonym (that
would mean a bug), and it **drops** any row whose text contains a known raw
Discord id or mention markup rather than publishing it.
=======
Three guards: the export **aborts** if an author field is not a pseudonym (that
would mean a bug), and it **drops** any row whose text contains a known raw
Discord id or mention markup, or that matches the [content blocklist](#content-filter),
rather than publishing it.
>>>>>>> theirs

## Configuration

Copy `.env.example` to `.env`. The only thing the bot strictly needs is
`BABBLE_DISCORD_TOKEN` (create the app at
<https://discord.com/developers/applications> and enable the **Message Content**
<<<<<<< ours
privileged intent). Everything else has a working default.
=======
privileged intent). Everything else has a working default, including the
[training feed](#training-feed) (`BABBLE_LOG_WEBHOOK_URL`) and the
[content blocklist](#content-filter) (`BABBLE_BLOCKLIST_PATH`), both of which
are entirely optional.
>>>>>>> theirs

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
<<<<<<< ours
  store.py       the corpus of triples, pseudonymous on disk
  exchanges.py   what it said and to whom, so corrections find their target
  export_hf.py   dataset + card, consent re-checked, identifiers guarded
=======
  blocklist.py   content filter -- normalisation + word-boundary matching
  blocklist.txt  the (small, starter) list itself
  store.py       the corpus of triples, pseudonymous on disk
  exchanges.py   what it said and to whom, so corrections find their target
  export_hf.py   dataset + card, consent and blocklist re-checked, ids guarded
  discord_feed.py training progress -> a Discord webhook, best-effort
>>>>>>> theirs
  logs.py        append-only structured + prose event log
  stats.py       snapshot and loss curve rendering
  cli.py         `babble <command>`
```

`core.py` is deliberately Discord-free: it takes plain dataclasses and returns
`Reply` objects. That is why the consent gate, correction capture, 👍 handling
and purge are all tested against a fake gateway, with no token and no network.

## Licence

MIT.
