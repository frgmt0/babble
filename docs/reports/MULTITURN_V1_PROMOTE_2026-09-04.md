# Multi-turn v1 promotion on jason's desktop (2026-09-04)

Booper now serves the local `hf-booper-multiturn-v1` export, replacing
Story-v2. Runtime code was already deployed from main at
`f439043d66bb2b1e2b1abd12085a546fa4f858f6` (PR #44). No model or dataset was
published to Hugging Face during this promotion.

## Training and quality gate

Mac run `runs/multiturn-v1` continued Story-v2 using
`configs/sft/multiturn-mac.json`: 366 steps, 11,179,465 processed tokens,
learning rate 2e-5, sequence length 1024. Finished September 4 at 17:34 PDT.
The example mix was 55% Discord, 10% TinyStories, 15% WritingPrompts,
10% no_robots, and 10% SmolTalk. Discord supplied 98,788 training targets
from conversation-group splits; 55 duplicate groups were dropped. Dataset
revisions and the split signature are recorded in local `quality.json`.

| Validation view | Baseline | Final |
| --- | ---: | ---: |
| Combined | 1.784995 | 1.771980 |
| Discord multi-turn | 2.561224 | 2.458014 |
| Discord overall | 2.594684 | 2.508892 |
| TinyStories | 1.087306 | 1.079328 |
| WritingPrompts | 2.208504 | 2.205860 |
| no_robots | 1.891561 | 1.885120 |
| SmolTalk | 1.448660 | 1.432523 |

Final/best step 366 passed the source gate. The largest cross-format
single-turn regression was +0.005582 nats (no_robots), below the 0.05 limit.
Multi-turn Discord loss fell about 4.0%; this measures likelihood, not a
4% increase in conversational accuracy. Custom INT8 export round-trip passed.

## Synthetic conversational review

Compared both exports through the desktop's live Python environment and
serving implementation, in an isolated source directory. Six fixed synthetic
cases, seed 20260904, best-of-four, 192-token generation ceiling. Story-v2
received its legacy current-message prompt; v1 received the trained role
transcript, including supplied history for follow-ups.

- With history naming a cat Pickles, v1 answered `Your cat's name is Pickles.`
  Story-v2, which received no history, invented Myrtle.
- After a pizza recommendation, v1 stayed on pizza but called it a healthy
  breakfast and failed to suggest a topping. Story-v2 said to get it from a shop.
- A project-completion follow-up produced a relevant but generic supportive
  reply from v1; Story-v2 drifted into a childhood story.
- Both produced weak greetings and stories with continuity problems.
- Without history, v1 invented a cat/dog named Lily. This is still a small,
  unreliable model; the promotion does not establish factual reliability.

These samples support activating the new context format, with modest quality
expectations. The old/new comparison measures the full intended migration;
it does not isolate training gains from the benefit of supplying history.
No synthetic messages were posted to Discord. Existing runtime tests cover
reply-chain selection, consent, identity isolation, persistence and forgetting;
no human Discord reply exchange was performed in this promotion.

## Performance

One warm staged `/bench`-equivalent sample per model, four CPU threads:

| Model | Actual candidate tokens | End-to-end aggregate TPS | Selected TPS | TTFT |
| --- | --- | ---: | ---: | ---: |
| Story-v2 | 36 / 64 / 64 / 64 | 131.1 | 36.8 | 52.8 ms |
| Multi-turn v1 | 64 / 64 / 64 / 64 | 147.7 | 36.9 | 56.9 ms |

V1 steady aggregate decode was 150.4 TPS; elapsed 1.733 s, average CPU 3.93
cores, RSS approximately 1160 MiB. Different candidate lengths explain the
aggregate TPS difference; this is not evidence of an additional 13% engine
speedup. See [the inference report](HF_INFERENCE_BENCHMARK_2026-09-04.md)
for controlled optimization measurements. These are bounded diagnostic runs,
not an established hardware maximum.

## Artifact integrity and activation

All six export files matched SHA256 after transfer. Principal hashes:

```
a60a88e2641d1650824ff943bb10a5271166b6fe9091d30a749dc0a04830b6be  model-int8.safetensors
a7a2e84a24af6828aa4b7b27220ea5a65bca222846bb512f019f1fe1804edf41  config.json
b1e1233481d5b2c3637fe8bcef81ec696fec835341680f0a0a72449bd699717a  tokenizer.json
```

The tokenizer is unchanged from Story-v2. Export size: 164,689,764 bytes.
The live environment now sets:

```
BABBLE_HF_MODEL_DIR=/home/jason/babble-live/artifacts/hf-booper-multiturn-v1
BABBLE_CONVERSATION_CONTEXT=1
BABBLE_CONVERSATION_MAX_TURNS=3
BABBLE_CONVERSATION_MAX_TOKENS=512
BABBLE_CONVERSATION_MAX_CHARS=0
BABBLE_MAX_NEW_TOKENS=510
```

Token-only prompt fitting matches training; 510 generated tokens plus the
512-token prompt and BOS/SEP fit the trained 1024-position sequence envelope.
Context follows explicit replies to the bot, for the same user/channel/guild,
and requires correction and corpus consent. New conversations start fresh.
Existing auto-training controls and other sampling settings were preserved.

At 2026-09-05 01:07:06 UTC (September 4, 18:07 PDT), the restarted service
logged the correct model directory, 149,602,432 parameters and CPU device.
It synced `bench` and reached `bot.ready` at 01:07:09 UTC: five guilds,
78.8 ms gateway latency. Running-process settings matched activation above.
Read-only Discord API verification confirmed the `booper` application and
`bench` command, Manage Server permission (32), DMs disabled. No error events
appeared after the model load during the immediate verification. Both
`babble-bot` and the separate `babble-boopit` service were active.

## Rollback

The previous complete environment and Story-v2 artifact directory were copied
to `~/babble-live/backups/multiturn-v1-promote-2026-09-04/` before activation.
The original Story-v2 artifact directory also remains in place. Backup files
are private runtime state, locally excluded from Git.

To restore the previous model and its matching legacy context settings on desktop:

```sh
cp ~/babble-live/backups/multiturn-v1-promote-2026-09-04/.env ~/babble-live/.env
systemctl --user restart babble-bot
```

Verify Story-v2 `model.load` followed by `bot.ready` after rollback. This does
not undo the independent inference optimizations or `/bench` code.
