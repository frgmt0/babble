import json
import random

import pyarrow.parquet as pq

random.seed(4)
W = "/tmp/maxperf/w4/"
d = pq.read_table(W + "hfcache/datasets--mookiezi--Discord-Dialogues/snapshots/a8b2294bd5b4acfe4ce537b688e7eee111c50fe2/data/train.parquet")
print(d.schema, d.num_rows)
idx = random.sample(range(d.num_rows), 600)
col = d.column("text")
with open(W + "discord.jsonl", "w") as f:
    for i in idx:
        f.write(json.dumps({"text": col[i].as_py()}) + "\n")
u = pq.read_table(W + "hfcache/datasets--HuggingFaceH4--ultrachat_200k/snapshots/8049631c405ae6576f93f445c6b8166f76f5505a/data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet")
print(u.schema.names, u.num_rows)
idx = random.sample(range(u.num_rows), 300)
m = u.column("messages")
with open(W + "ultrachat.jsonl", "w") as f:
    for i in idx:
        f.write(json.dumps({"messages": m[i].as_py()}) + "\n")
