# QA eval: longctx-v1

- model dir: `/home/jason/babble-live/artifacts/hf-booper-longctx-v1`
- runtime: native, decoding: greedy (top_k=1, best_of=1), max_new_tokens 64
- prompt format: role_transcript_v1, items: 250, wall: 9s

**overall accuracy 13.3%** (micro, answerable items) | macro 15.5% | clean (no flag) 65.6%

| category | n | acc % | echo % | empty % | non_answer % | repetitive % | clean % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fact_common | 70 | 7.1 | 24.3 | 0.0 | 15.7 | 0.0 | 60.0 |
| math | 40 | 2.5 | 40.0 | 0.0 | 5.0 | 0.0 | 55.0 |
| definition | 30 | 6.7 | 70.0 | 0.0 | 0.0 | 0.0 | 30.0 |
| about_bot | 25 | 4.0 | 0.0 | 0.0 | 16.0 | 0.0 | 84.0 |
| commonsense | 35 | 28.6 | 5.7 | 0.0 | 8.6 | 0.0 | 85.7 |
| followup | 25 | 44.0 | 4.0 | 0.0 | 8.0 | 0.0 | 88.0 |
| chat | 25 | - | 12.0 | 0.0 | 16.0 | 0.0 | 72.0 |
| **all** | 250 | 13.3 | 24.0 | 0.0 | 10.4 | 0.0 | 65.6 |
