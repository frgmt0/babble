# QA eval: qa-v1

- model dir: `~/babble-live/artifacts/hf-booper-qa-v1`
- runtime: native, decoding: greedy (top_k=1, best_of=1), max_new_tokens 64
- prompt format: role_transcript_v1, items: 250, wall: 9s

**overall accuracy 24.0%** (micro, answerable items) | macro 28.5% | clean (no flag) 92.0%

| category | n | acc % | echo % | empty % | non_answer % | repetitive % | clean % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fact_common | 70 | 10.0 | 4.3 | 0.0 | 0.0 | 0.0 | 95.7 |
| math | 40 | 2.5 | 12.5 | 0.0 | 0.0 | 0.0 | 87.5 |
| definition | 30 | 63.3 | 10.0 | 0.0 | 0.0 | 0.0 | 90.0 |
| about_bot | 25 | 32.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| commonsense | 35 | 31.4 | 2.9 | 0.0 | 2.9 | 0.0 | 94.3 |
| followup | 25 | 32.0 | 4.0 | 0.0 | 0.0 | 0.0 | 96.0 |
| chat | 25 | - | 12.0 | 0.0 | 12.0 | 0.0 | 76.0 |
| **all** | 250 | 24.0 | 6.4 | 0.0 | 1.6 | 0.0 | 92.0 |
