# QA eval: longctx-v1-sampled

- model dir: `/home/jason/babble-live/artifacts/hf-booper-longctx-v1`
- runtime: native, decoding: sampled (seed 1234+i), max_new_tokens 64
- prompt format: role_transcript_v1, items: 250, wall: 17s

**overall accuracy 13.3%** (micro, answerable items) | macro 15.0% | clean (no flag) 66.0%

| category | n | acc % | echo % | empty % | non_answer % | repetitive % | clean % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fact_common | 70 | 10.0 | 22.9 | 0.0 | 15.7 | 0.0 | 61.4 |
| math | 40 | 0.0 | 35.0 | 0.0 | 12.5 | 2.5 | 50.0 |
| definition | 30 | 13.3 | 60.0 | 0.0 | 0.0 | 0.0 | 40.0 |
| about_bot | 25 | 12.0 | 4.0 | 0.0 | 16.0 | 0.0 | 80.0 |
| commonsense | 35 | 22.9 | 11.4 | 0.0 | 14.3 | 0.0 | 74.3 |
| followup | 25 | 32.0 | 4.0 | 0.0 | 8.0 | 0.0 | 88.0 |
| chat | 25 | - | 4.0 | 0.0 | 8.0 | 0.0 | 88.0 |
| **all** | 250 | 13.3 | 22.0 | 0.0 | 11.6 | 0.4 | 66.0 |
