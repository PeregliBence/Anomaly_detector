"""Does hyperparameter tuning help? Nested tuning against the fixed defaults.

At every annual refit, each candidate setting is scored on purged splits of
that fold's training window only, and the best one is used for the next
year. The result is compared with the final evaluation's fixed defaults under
the same protocol. The grids are deliberately small: the question is whether
tuning matters at all, and each fold multiplies the fitting cost by
grid size times inner splits.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import experiment, models, store
from src.config import setup_logging

warnings.filterwarnings("ignore")
setup_logging("WARNING")

PROTOCOL = dict(refit_every=252, train_stride=5)
GRIDS = {
    "lightgbm": [{"num_leaves": nl, "min_child_samples": mcs}
                 for nl in (15, 63) for mcs in (100, 1000)],
    "xgboost": [{"max_depth": d, "min_child_weight": mcw}
                for d in (3, 7) for mcw in (50, 500)],
}


def main() -> None:
    pd.set_option("display.width", 200)
    models.PARAM_GRIDS.update(GRIDS)
    t0 = time.time()
    panel = experiment.load_sharadar_panel(horizons=[5, 20], seq_len=20)
    alt = [c for c in panel.columns if c.startswith("alt_")]
    panel = panel.drop(columns=alt)

    fixed = store.experiments()
    fixed = fixed[fixed["name"] == "final"].set_index(["model", "horizon"])["ic_mean"]

    rows = []
    for h in (5, 20):
        for model in GRIDS:
            # Resume an interrupted run: anything already stored is kept.
            if store.find("tuned", model, h) is not None:
                print(f"tuned {model} h={h}: already stored, skipping", flush=True)
                continue
            t = time.time()
            preds, sc = experiment.run(panel, horizon=h, model_names=[model],
                                       tune=True, tune_splits=2, **PROTOCOL)
            if sc.empty:
                continue
            metrics = sc.iloc[0].drop(["model", "horizon", "target"]).to_dict()
            store.save_experiment("tuned", model, h, "cross_sectional", "point_in_time",
                                  {**PROTOCOL, "seq_len": 20, "tuned": True,
                                   "grid": GRIDS[model], "inner_splits": 2},
                                  preds, metrics)
            chosen = pd.DataFrame(preds.attrs.get("fold_params", []))
            rows.append({"horizon": h, "model": model, "ic_tuned": metrics["ic_mean"],
                         "ic_fixed": fixed.get((model, h)), "minutes": round((time.time() - t) / 60)})
            print(rows[-1], flush=True)
            if not chosen.empty:
                print("  parameters chosen per fold:\n" + chosen.to_string(index=False), flush=True)

    out = pd.DataFrame(rows)
    out["gain"] = out["ic_tuned"] - out["ic_fixed"]
    Path("results/tuning").mkdir(parents=True, exist_ok=True)
    out.to_csv("results/tuning/summary.csv", index=False)
    print("\n" + out.round(4).to_string(index=False))
    print(f"\nTotal {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
