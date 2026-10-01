# QA eval baseline — 2026-10-01

A held-out question-answering eval for booper (`bench/qa/`), and its first
numbers on the live model (`hf-booper-longctx-v1`) and two earlier exports.
Nothing was promoted or changed on the live box; the model directories were
only read.

## What the eval is

`bench/qa/questions.jsonl` has 250 hand-written items. None of them come from
an HF dataset, so the set can stay out of the training mix.

| category | n | scored by |
| --- | ---: | --- |
| fact_common | 70 | answer alias (capitals, colours, days/months, animals, basic science, famous people) |
| math | 40 | the number, written as digits or words (single-step + − × ÷, phrased several ways) |
| definition | 30 | ≥1 keyword out of 3–8 (a keyword of 4+ letters also matches as a word prefix) |
| about_bot | 25 | "booper" / "bot", or yes/no where the question is polar; `must_not` catches "i'm human"-style claims |
| commonsense | 35 | yes/no (polar) or answer alias |
| followup | 25 | answer alias; `history` sets up a fact earlier in the conversation and the question refers back to it |
| chat | 25 | no answer key; only the failure flags apply |

`bench/qa/run_eval.py` loads an export through `babble.hfserve.make_generator`,
which is the same generator the bot uses (`hf` backend, `native` runtime by
default). It builds the prompt the way `Babble._generation_prompt` does: if the
export's `config.json` says `babble_prompt_format: role_transcript_v1`, the
backend's own `conversation_prompt` writes a `user:/assistant:` transcript
(turns = `babble_history_turns`, 512 prompt tokens, no character cap, which
matches the live `.env`). Otherwise the model gets the bare question.

Decoding uses the live sampler settings (temperature 0.5, top_k 40, top_p 0.9,
repetition_penalty 1.15, no_repeat_ngram_size 4, best_of 4,
frequency/presence penalties off), with `max_new_tokens` set to 64. The default
mode is `--greedy`, which sets top_k=1 and best_of=1 in that same sampler: it
takes the argmax after the production repetition penalty and n-gram ban, so
it is deterministic. `--sampled` runs the full live sampler with
`torch.manual_seed(seed+i)` before item i. Two `--sampled` runs gave identical
output on all 250 items. The prefix KV cache and speculative decoding are off,
since spec decoding does not change the output distribution.

Scoring (`bench/qa/scoring.py`, unit-tested in `tests/test_qa_eval.py`):
before matching, responses are lowercased, stripped of punctuation, and number
words are converted to digits. An answer counts only as whole tokens ("12"
does not match "112" or "12.5"). Polar items are judged on the first yes/no
word in the response. A `must_not` hit always makes the item wrong. Each item
also gets four failure flags:

* `echo`: the response is a substring of the question, or ≥60% of its distinct
  tokens are in the question. A correct response is never flagged as echo.
* `empty`
* `non_answer`: a short "idk"-style dodge, or a reply made only of questions
* `repetitive`: distinct-token ratio below 0.5, or some 3-gram repeated ≥3 times

**Overall accuracy** is the micro-average over the 225 items with an answer
key. **Clean** is the share of all 250 items with no flag.

Speed: the whole set takes 9–15 s per model on 4 threads of the i7-4790
(median 29–39 ms per item, greedy).

## Results (greedy, 250 items)

| model | prompt format | overall acc | macro acc | clean | echo | non_answer | repetitive | empty |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **longctx-v1 (live)** | role_transcript_v1, 8 turns | **13.3%** | 15.5% | 65.6% | 24.0% | 10.4% | 0.0% | 0.0% |
| multiturn-v1 | role_transcript_v1, 3 turns | 20.4% | 24.0% | 72.4% | 14.8% | 12.8% | 0.0% | 0.0% |
| story-v2 | raw single-turn | 18.2% | 21.0% | 75.2% | 10.8% | 14.0% | 0.0% | 0.0% |
| longctx-v1, `--sampled` (seed 1234) | role_transcript_v1 | 13.3% | 15.0% | 66.0% | 22.0% | 11.6% | 0.4% | 0.0% |

Accuracy by category (%):

| category | longctx-v1 (live) | multiturn-v1 | story-v2 | longctx-v1 sampled |
| --- | ---: | ---: | ---: | ---: |
| fact_common (70) | 7.1 | 11.4 | 5.7 | 10.0 |
| math (40) | 2.5 | 0.0 | 0.0 | 0.0 |
| definition (30) | 6.7 | 40.0 | 53.3 | 13.3 |
| about_bot (25) | 4.0 | 8.0 | 20.0 | 12.0 |
| commonsense (35) | 28.6 | 28.6 | 42.9 | 22.9 |
| followup (25) | 44.0 | 56.0 | 4.0* | 32.0 |

\* story-v2 is a single-turn model. The bot would serve it without history, so
it never sees the fact a followup item asks about. Its followup score
reflects the prompt format, not its memory.

Echo rate by category for the live model: definition 70%, math 40%,
fact_common 24%, chat 12%, commonsense 6%, followup 4%, about_bot 0%.

Full per-item output: `bench/qa/results/<label>.json`, with a per-model summary
in `<label>.md`.

## Observations (live model)

* **It mostly restates the question instead of answering it.** It answers 21
  of 30 definition questions by repeating the noun ("what is a dog" →
  "A dog"). Capital-city questions come back as "The capital is France." It
  answered none of the 12 capitals correctly.
* **Math is effectively 0.** The one "correct" item is a lucky match (below).
  Typical outputs copy the operands ("7+5", "50 +", "10 times").
* **It never names itself.** None of the 250 responses contain "booper", except
  "I am not a booper" to a question that already said "booper". In about_bot,
  10 of 25 answers claim to be a person ("I am a human", "I am a man",
  "I'm a girl").
* **"Idk" is a frequent fallback.** 28 of 250 responses open with
  "idk" / "I don't know".
* **Followup is its strongest category (44%).** When the earlier user turn
  states the fact ("my favourite colour is purple"), it often repeats it back
  correctly.
* **It never loops.** Repetition is 0% under greedy decoding with the live
  n-gram ban, and 0.4% when sampled. Replies are short (6.1 words on
  average).
* **The older exports score higher on this set.** multiturn-v1 and story-v2
  do much better on definition (40% and 53% vs 6.7%) and echo less. The live
  longctx-v1 gives shorter, chattier replies, and that style hurts it here.

## Representative live responses (greedy)

| id | question | response | scored |
| --- | --- | --- | --- |
| fact_common-016 | what colour is the sky on a clear day? | The sky is blue | correct |
| fact_common-017 | what color is grass | green | correct |
| fact_common-051 | what do bees make? | Bee honey, honey | correct |
| definition-005 | what does a doctor do | A doctor is a medical professional who diagnoses and treats cancer. … | correct |
| commonsense-029 | which direction does the sun rise in? | The sun rises in the east, and the west is in front of it. … | correct |
| followup-002 | (history: "my favourite colour is purple") what's my favourite colour | purple | correct |
| followup-013 | (history: "let's talk about the ocean") what are we talking about | the ocean | correct |
| chat-001 | how's your day going? | I'm good wbu | clean |
| fact_common-001 | what is the capital of france? | The capital is France. | wrong, echo |
| fact_common-002 | capital of japan? | Idk | wrong, non_answer |
| fact_common-030 | how many hours are in a day? | I think it's like 5 minutes | wrong |
| math-004 | what is 2 plus 2? | 2 + 2 = 2 | wrong, echo |
| definition-003 | what is a dog | A dog | wrong, echo |
| about_bot-008 | are you a bot? | I am a human | wrong |
| about_bot-002 | what's your name | Idk | wrong, non_answer |
| commonsense-005 | can dogs fly | yes | wrong |
| followup-022 | (history: capital of france → "paris") and what about germany? | paris | wrong |
| chat-020 | thanks for chatting with me | I am not a fan of the game | clean (no flag) |

Scorer limitations, seen in this run:

* `math-034` "what is 63 divided by 9" → "6 divided = 7" is scored correct.
  The digit is there, but the reasoning is not.
* `commonsense-006` "can pigs fly?" → "pigs are not a type of animals" is
  scored correct, because "not" is the first polarity word.
* `fact_common-022` "what color are strawberries" → "a mix of red and blue"
  passes, because hedged multi-answers are not penalised.
* Affirmations without a yes-word ("is fire hot?" → "I think it is") score
  wrong. That happened on 3 polar items in the live run.

These errors go in both directions and are rare compared with the gaps
between models, but the absolute numbers are approximate to within a few
points.

## Reproduce

```bash
uv venv && uv pip install -e ".[dev,hf]"
.venv/bin/python bench/qa/run_eval.py --model-dir ~/babble-live/artifacts/hf-booper-longctx-v1 --label longctx-v1
.venv/bin/python bench/qa/run_eval.py --model-dir ~/babble-live/artifacts/hf-booper-longctx-v1 --label longctx-v1-sampled --sampled
.venv/bin/python -m pytest -q tests/test_qa_eval.py
```

The eval only reads the model directory. It builds `Settings.for_root` on a
temp dir, so it never touches a data dir or `.env`.
