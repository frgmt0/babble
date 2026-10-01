"""Screen error + margin statistics for the int4 two-stage head (track w4).

For the int4 group-wise screen of the tied head it reports, over the captured
real hidden states:
* the screen error |s'_r - s_r| (max per position, and quantiles);
* the margin m = (exact 40th-best post-penalty candidate score) - (best screen
  score among non-candidates). A runtime rule "fall back to the full exact head
  when m < delta" is exact whenever every non-candidate's screen error is
  below delta; the table shows the fallback rate for a few deltas next to the
  observed max error.

    python bench/extreme/w4_head_margin.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.extreme.w4_head_study import K, penalize  # noqa: E402
from bench.extreme.w4_quant_study import W, int8_fp32, load_packed, rtn  # noqa: E402

torch.set_num_threads(int(os.environ.get("W4_THREADS", "3")))


def main():
    packed = load_packed()
    Wt = int8_fp32(packed, "model.embed_tokens.weight")
    V = Wt.shape[0]
    hidden = torch.load(W / "ref-hidden.pt")
    hs = torch.stack([h for h, _ in hidden]).float()
    hists = [ids for _, ids in hidden]
    n = hs.shape[0]
    exact = hs @ Wt.T
    out = []
    for g in (32, 64, 128):
        Wq = rtn(Wt, 4, g, True)
        sc = hs @ Wq.T
        err = (sc - exact).abs()
        maxerr = err.max(dim=1).values
        row = {"screen": f"int4-g{g}", "err_max": float(maxerr.max()),
               "err_p99_of_rowmax": float(maxerr.quantile(0.99)), "err_median_of_rowmax": float(maxerr.median()),
               "err_rms": float(err.pow(2).mean().sqrt())}
        for N in (128, 256, 384):
            margins = []
            for i in range(n):
                s_pen, mask = penalize(exact[i], hists[i])
                scr = sc[i].masked_fill(mask, -float("inf"))
                top = scr.topk(N + 1)
                cand = mask.clone()
                cand[top.indices[:N]] = True
                finite = torch.isfinite(s_pen)
                cs = s_pen.masked_fill(~cand | ~finite, -float("inf"))
                kc = cs.topk(K).values[-1]
                margins.append(float(kc - top.values[N]))
            m = torch.tensor(margins)
            row[f"N{N}"] = {
                "margin_min": float(m.min()),
                "margin_p01": float(m.quantile(0.01)),
                **{f"fallback@{d}": float((m < d).float().mean()) for d in (0.25, 0.5, 0.75, 1.0, 1.5)},
            }
        print(json.dumps(row), flush=True)
        out.append(row)
    with open(W / "head_margin.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
