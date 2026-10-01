# QA eval: story-v2

- model dir: `/home/jason/babble-live/artifacts/hf-booper-story-v2`
- runtime: native, decoding: greedy (top_k=1, best_of=1), max_new_tokens 64
- prompt format: raw (single-turn), items: 250, wall: 15s

**overall accuracy 18.2%** (micro, answerable items) | macro 21.0% | clean (no flag) 75.2%

| category | n | acc % | echo % | empty % | non_answer % | repetitive % | clean % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fact_common | 70 | 5.7 | 15.7 | 0.0 | 8.6 | 0.0 | 75.7 |
| math | 40 | 0.0 | 27.5 | 0.0 | 10.0 | 0.0 | 62.5 |
| definition | 30 | 53.3 | 10.0 | 0.0 | 0.0 | 0.0 | 90.0 |
| about_bot | 25 | 20.0 | 0.0 | 0.0 | 32.0 | 0.0 | 68.0 |
| commonsense | 35 | 42.9 | 2.9 | 0.0 | 11.4 | 0.0 | 85.7 |
| followup | 25 | 4.0 | 0.0 | 0.0 | 40.0 | 0.0 | 60.0 |
| chat | 25 | - | 4.0 | 0.0 | 12.0 | 0.0 | 84.0 |
| **all** | 250 | 18.2 | 10.8 | 0.0 | 14.0 | 0.0 | 75.2 |
