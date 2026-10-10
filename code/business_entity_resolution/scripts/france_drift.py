"""Per-country drift guard between two final candidates (same candidate rows): matches per S1, uncertain pairs per S1
and the share of S1s whose decided match set changes. Label-free; used to block a model update that moves the country
absent from train (France) much more than the labelled countries.

    python scripts/france_drift.py --base runs/E015/sub_E033_e030_c383 --new runs/E015/sub_E034_ow04 \
        --t-seen 0.75 --t-unseen 0.85
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.er_data import cache_dir  # noqa: E402
from src.er_decide import decide  # noqa: E402


def decisions(p: pd.DataFrame, unseen: pd.Series, t_seen: float, t_unseen: float) -> dict:
    u = p.s1_id.map(unseen).fillna(False).to_numpy()
    m = decide(p[~u], t_seen, assign=True)
    m.update(decide(p[u], t_unseen, assign=True))
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--t-seen", type=float, default=0.75)
    ap.add_argument("--t-unseen", type=float, default=0.85)
    ap.add_argument("--max-flip", type=float, default=0.03, help="max share of unseen-country S1s whose set changes")
    ap.add_argument("--max-shift", type=float, default=0.02, help="max relative change of unseen matches per S1")
    a = ap.parse_args()
    c = cache_dir()
    s1 = pd.read_parquet(c / "test_s1.parquet", columns=["entity_id", "country"]).set_index("entity_id").country
    seen = set(pd.read_parquet(c / "train_s1.parquet", columns=["country"]).country.unique())
    unseen = ~s1.isin(seen)
    pb = pd.read_parquet(Path(a.base) / "test_probs_stage2.parquet")
    pn = pd.read_parquet(Path(a.new) / "test_probs_stage2.parquet")
    same_rows = len(pb) == len(pn) and pb[["s1_id", "cand_id"]].merge(pn[["s1_id", "cand_id"]]).shape[0] == len(pb)
    mb, mn = decisions(pb, unseen, a.t_seen, a.t_unseen), decisions(pn, unseen, a.t_seen, a.t_unseen)
    out, ok = {"same_candidate_rows": same_rows}, same_rows
    for ctry in sorted(s1.unique()):
        ids = s1.index[s1 == ctry]
        kb = sum(len(mb.get(i, ())) for i in ids) / len(ids)
        kn = sum(len(mn.get(i, ())) for i in ids) / len(ids)
        flip = sum(set(mb.get(i, ())) != set(mn.get(i, ())) for i in ids) / len(ids)
        ub = pb[pb.s1_id.isin(set(ids)) & (pb.prob >= 0.3) & (pb.prob < 0.99)].shape[0] / len(ids)
        un = pn[pn.s1_id.isin(set(ids)) & (pn.prob >= 0.3) & (pn.prob < 0.99)].shape[0] / len(ids)
        out[ctry] = {"matches_per_s1": [round(kb, 4), round(kn, 4)], "uncertain_per_s1": [round(ub, 4), round(un, 4)],
                     "s1_changed": round(flip, 4), "unseen": ctry not in seen}
        if ctry not in seen and (flip > a.max_flip or abs(kn / kb - 1) > a.max_shift):
            ok = False
    out["DRIFT_GUARD"] = "PASS" if ok else "FAIL"
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
