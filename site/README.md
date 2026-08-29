# site/ — booper.frgmt.xyz

Cloudflare Worker with static assets. Serves the founding-documents pages plus
`/runs`, a live training dashboard backed by a small `/api/runs` KV feed that
`sft/sft_longform.py`'s `Reporter` POSTs metrics to.

## Local dev

```
cd site
npx wrangler dev --local
```

Put `RUNS_TOKEN=<any-string>` in a `site/.dev.vars` file first (gitignored) —
that's the bearer token `/api/runs/<name>` POSTs must present locally. Feed it
fake records with curl and open `http://localhost:8787/runs`.

## Deploying `/runs` for the first time

1. Create the KV namespace and paste its id into `wrangler.jsonc`:
   ```
   npx wrangler kv namespace create RUNS
   ```
   Replace `REPLACE_WITH_KV_NAMESPACE_ID` under `kv_namespaces` with the id it
   prints.
2. Set the production secret (the same value the trainer will send):
   ```
   npx wrangler secret put RUNS_TOKEN
   ```
3. Deploy:
   ```
   npx wrangler deploy
   ```

## Pointing the trainer at it

On the training machine, add to `.env.sft`:

```
BABBLE_RUNS_URL=https://booper.frgmt.xyz
BABBLE_RUNS_TOKEN=<same value you put in RUNS_TOKEN>
```

`sft_longform.py` only reports to `/runs` when both are set; without them it
just writes `runs/<name>/metrics.jsonl` locally as usual.
