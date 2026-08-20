## Booper regression investigation

### Root cause

The model did get trained, but not on the same kind of data as the "better"
2026-08-17 checkpoint. The apparent regression came from the corpus-only pivot
in `11955bd` ("Train on the collected corpus only"), which removed the old
story/dictionary stage-1 pretrain code path. That left stage 1 as random-init
training on the consented Discord corpus alone.

Evidence:

- The "good" 2026-08-17 samples came from the old story-base plus the user-text
  pass, not from a larger corpus-only run.
- `babble/pretrain.py` and the external-corpus path were removed in `11955bd`,
  about four hours before the `pretrained.pt` file later written on 2026-08-17.
- Today's corpus-only trainer still improves as rows increase: on the held-out
  validation metric, live runs moved from `2.9830` to `2.8444` to `2.7872` and
  then, after the final retrain below, to `2.7507`.
- So "more data made it worse" was the wrong comparison. The real comparison is
  "vastly less total text, but more user rows": story+voice had a much larger
  text base and therefore looked more word-shaped even though the newer
  corpus-only run kept improving within its own regime.

### What actually triggers training

Training is triggered only when the stored corpus has grown by
`BABBLE_TRAIN_TRIGGER_ROWS` new rows since the last run (default `100`); the
running bot notices that threshold crossing and launches one detached
`babble train` subprocess, while `babble-train.service` being disabled means
there is no separate always-on training loop.

### Comparable metrics

`loss.jsonl` now records comparable checkpoint metrics:

- `loss`: train loss for that checkpoint window
- `val_loss`: held-out validation loss used to pick the served checkpoint
- `stored_rows`: raw corpus row count used by the +N-row trigger
- `train_rows` / `val_rows`: the actual split sizes for that run

`train_state.json` now also distinguishes:

- `step`: the best-val checkpoint persisted to `latest.pt`
- `steps_run`: how far the loop actually ran before early stop
- `last_trained_rows`: the stored corpus count that satisfied the trigger

### Before/after on live data

| moment | best step served | steps run | stored rows | train/val rows | val_loss |
|---|---:|---:|---:|---:|---:|
| before final retrain | 300 | unknown in persisted state | 394 | 314 / 79 | 2.7872 |
| after final retrain | 200 | 350 | 400 | 320 / 80 | 2.7507 |

The final retrain persisted:

```json
{
  "last_trained_rows": 401,
  "step": 200,
  "at": "2026-08-20T06:18:38+00:00",
  "steps_run": 350
}
```

One new row landed during the run, so `train_state.json` shows `401` while the
best checkpoint lines in `loss.jsonl` still show `stored_rows: 400`.

Immediately before that forced retrain, `babble train-status` on the live install
reported `corpus rows: 400 · last trained at: 394 rows · new since: 6 · threshold: 100 · due: no`,
which is the direct answer to "did it not get trained?": it had trained, and it
was simply not due again yet.

### Required probe samples

Before final retrain (`latest.pt` at step 300):

- `hello` -> ` mer t w chede t banou med cous y w do t t the the we me ge lg y the dore be t the s y thas t s t the t t s s as be the s at ase. bete theng age s bot o rend ghedon n a thathe? tse be weas buthhe then as t wed chit d can the matouthe t as athe sushe mastha`
- `the cat` -> `or wh t re cthe that whe an yowhere t s d t bus the te t n th d s ithe poro than the che d the n it ms thar le s n t a y t at to anthe moree le t thad nlit she d te atithhe the mtse the e tha s se do t ct in s the t thin t de d bad the ases nge in fus as l`
- `where` -> ` is a t t the t an s a i so y ate t cke s w de s then acore ore n thee ashat hat an t athanore le s at t so r t s ttt an in t se be the be mee thane ithe an an e t eas lan otore be it wat whe b then athe mne a whe t thane as athoure me d t ere s the we tes`
- `what is love` -> ` t at the an ye t g t ane thed t the be s t rengon te the thas whe ma o and whe the anthe the nthe sis fus me ang s me s the the mhe t le yol me a the at we athe e she the whe  y m fus mithen we ithus t the t the s as athhe bate t athe lin as an wathe be f`

After final retrain (`latest.pt` at step 200):

- `hello` -> `nbe the whe t meser t t indes wen ing it as wen or d t me it`
- `the cat` -> ` b t bed ithiny thar ing whe we men in th the ha the bu thacon athandoucthithe t fe it wherts isere the the an math wathatharot itonghinthol in`
- `where` -> ` ith y in`
- `what is love` -> ` g in st at is t in t w inong we thathalthindhe ise at at a. me we that at it tiu t at whexe. ou l out s ing t ma at we che mut amelouthe ithe as anont t at inke ors were t nthe whesoule wher wis it whaathind thes athe ot mthat me isedodtha ma whend bu ine`

### Tradeoff table

| setup | text base | best comparable val_loss | probes |
|---|---|---:|---|
| archived story+voice run | large external story/dictionary base + user text | 1.9770 | word-shaped, sometimes phrase-like |
| current corpus-only run | consented Discord corpus only | 2.7507 | still letter-soup, though less noisy than before |

### Bottom line

This was not "it never trained." It trained exactly when the +100-row trigger
said to train, but since `11955bd` it has been training from random init on a
few hundred Discord rows instead of inheriting the old large story base, so the
model remained much less word-like even as its corpus-only validation loss
improved.
