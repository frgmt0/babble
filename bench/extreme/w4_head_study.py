"""Two-stage lm_head screen study (track w4, phase 1b).

Uses the final hidden states captured by `w4_quant_study.py ref` (real-looking
chat sequences, random positions) and the int8 tied head. For each cheap
screen it measures, over thousands of positions:

* raw recall: true raw top-40 contained in the screen's top-N;
* live coverage: the true *post-penalty* top-40 (repetition 1.15, frequency
  0.12, no-repeat-ngram 4 over prompt+generated history, as the engine warps)
  contained in the candidate set = penalized tokens (computed exactly anyway)
  U screen top-N over the unpenalized tokens. When covered, the sampler sees
  exactly the same survivors and probabilities as with the full head;
* certified rate: with a per-row error bound |s'_r - s_r| <= b_r(h), the
  fraction of positions where max_{non-candidates}(s'_r + b_r) < the exact
  40th-best candidate score -- i.e. exactness is *proven* at runtime and the
  full head is only needed as a fallback for the rest.

    python bench/extreme/w4_head_study.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.extreme.w4_quant_study import W, int8_fp32, load_packed, rtn  # noqa: E402

torch.set_num_threads(int(os.environ.get("W4_THREADS", "3")))
K = 40
REP, FREQ, NGRAM = 1.15, 0.12, 4


def penalize(logits: torch.Tensor, hist: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    s = logits.clone()
    h = torch.tensor(hist)
    uniq, counts = torch.unique(h, return_counts=True)
    v = s[uniq]
    s[uniq] = torch.where(v < 0, v * REP, v / REP)
    s[uniq] -= counts.float() * FREQ
    n = NGRAM
    if len(hist) >= n:
        tail = hist[len(hist) - (n - 1):]
        for i in range(len(hist) - n + 1):
            if hist[i : i + n - 1] == tail:
                s[hist[i + n - 1]] = -float("inf")
    mask = torch.zeros(s.shape[0], dtype=torch.bool)
    mask[uniq] = True
    return s, mask


def main():
    packed = load_packed()
    Wt = int8_fp32(packed, "model.embed_tokens.weight")  # [V, H] exact head
    V, H = Wt.shape
    hidden = torch.load(W / "ref-hidden.pt")
    hs = torch.stack([h for h, _ in hidden]).float()
    hists = [ids for _, ids in hidden]
    n = hs.shape[0]
    print(f"{n} positions, mean history {sum(map(len, hists)) / n:.0f} tokens")
    exact = hs @ Wt.T  # [n, V]

    hn = hs.norm(dim=1, keepdim=True)

    def screens():
      # name, scores [n,V], bound [n,V], bytes
      for g in (32, 64, 128):
        for bits in (4, 3):
            Wq = rtn(Wt, bits, g, True)
            err = (Wt - Wq).norm(dim=1)  # per-row L2
            yield f"int{bits}-g{g}", hs @ Wq.T, hn * err.unsqueeze(0), V * H * bits / 8 + V * (H // g) * 2
      U, S, Vh = torch.linalg.svd(Wt, full_matrices=False)
      for r in (32, 64, 128, 256):
        A = U[:, :r] * S[:r]  # [V, r]
        B = Vh[:r]  # [r, H]
        # store A as int8 per row (what the engine would stream), B fp32 (tiny)
        Aq = rtn(A, 8, r, True)
        Wr = Aq @ B
        sc = (hs @ B.T) @ Aq.T
        # bound: residual rows R = W - Wr; |R_r . h| <= ||R_r|| ||h||
        # tighter for the SVD part: split h = P h + (I-P) h with P = B^T B
        resid = Wt - Wr
        h_perp = hs - (hs @ B.T) @ B
        Rp = Wt - (Wt @ B.T) @ B  # component of W rows outside the subspace
        # resid.h = (A - Aq)(B h) + Rp . h_perp ;  bound both by Cauchy-Schwarz
        e1 = (A - Aq).norm(dim=1)
        bound = (hs @ B.T).norm(dim=1, keepdim=True) * e1.unsqueeze(0) + h_perp.norm(dim=1, keepdim=True) * Rp.norm(dim=1).unsqueeze(0)
        del resid
        yield f"lowrank-r{r}-int8", sc, bound, V * r + V * 4 + r * H * 4

    results = []
    pen = [penalize(exact[i], hists[i]) for i in range(n)]
    for name, sc, bound, nbytes in screens():
        t = time.time()
        row = {"screen": name, "MB": round(nbytes / 1e6, 2), "positions": n}
        raw_top = exact.topk(K, dim=1).indices
        for N in (64, 128, 256, 512):
            top_n = sc.topk(N, dim=1).indices
            raw_rec = 0
            cover = 0
            cert = 0
            cand_sizes = 0
            for i in range(n):
                sel = torch.zeros(V, dtype=torch.bool)
                sel[top_n[i]] = True
                raw_rec += int(sel[raw_top[i]].all())
                s_pen, mask = pen[i]
                # candidates: penalized (exact anyway) + screen top-N among unpenalized
                scr = sc[i].masked_fill(mask, -float("inf"))
                cand = mask.clone()
                cand[scr.topk(N).indices] = True
                cand_sizes += int(cand.sum())
                finite = torch.isfinite(s_pen)
                kth = s_pen[finite].topk(min(K, int(finite.sum()))).values[-1]
                true_set = (s_pen >= kth) & finite
                cover += int(cand[true_set].all())
                # certificate (selection by upper bound would be tighter; we use the
                # same screen order and check the bound on the non-candidates)
                cs = s_pen.masked_fill(~cand | ~finite, -float("inf"))
                kc = cs.topk(K).values[-1]
                ub = (sc[i] + bound[i]).masked_fill(cand, -float("inf")).max()
                cert += int(ub < kc)
            row[f"N{N}"] = {
                "raw_recall": raw_rec / n,
                "live_cover": cover / n,
                "certified": cert / n,
                "mean_cand": cand_sizes / n,
            }
        row["s"] = round(time.time() - t, 1)
        print(json.dumps(row), flush=True)
        results.append(row)
    with open(W / "head_study.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
