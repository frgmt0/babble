# QA eval: multiturn-v1

- model dir: `/home/jason/babble-live/artifacts/hf-booper-multiturn-v1`
- runtime: native, decoding: greedy (top_k=1, best_of=1), max_new_tokens 64
- prompt format: role_transcript_v1, items: 250, wall: 14s

**overall accuracy 20.4%** (micro, answerable items) | macro 24.0% | clean (no flag) 72.4%

| category | n | acc % | echo % | empty % | non_answer % | repetitive % | clean % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fact_common | 70 | 11.4 | 17.1 | 0.0 | 11.4 | 0.0 | 71.4 |
| math | 40 | 0.0 | 32.5 | 0.0 | 20.0 | 0.0 | 47.5 |
| definition | 30 | 40.0 | 16.7 | 0.0 | 0.0 | 0.0 | 83.3 |
| about_bot | 25 | 8.0 | 0.0 | 0.0 | 24.0 | 0.0 | 76.0 |
| commonsense | 35 | 28.6 | 11.4 | 0.0 | 8.6 | 0.0 | 80.0 |
| followup | 25 | 56.0 | 0.0 | 0.0 | 4.0 | 0.0 | 96.0 |
| chat | 25 | - | 12.0 | 0.0 | 24.0 | 0.0 | 64.0 |
| **all** | 250 | 20.4 | 14.8 | 0.0 | 12.8 | 0.0 | 72.4 |
