"""Do the tree models memorise dates through the market-state features?

Market features (VIX, yield curve, credit spreads, index volatility...) take
the same value for every stock on a date, so together they identify the date.
A tree can then learn which stocks did well on that particular date — true in
the training data, useless afterwards. If that drives the gap between the
training and validation curves, removing those columns should make the
training loss fall much less while the validation loss stays where it was.

Same windows, split and settings as scripts/learning_curves.py; the rows are
added to its output under the model names `<model>_no_market`.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import learning_curves as lc
from src import experiment, models
from src.config import setup_logging
from src.neural import _time_split
from src.panel import split_xy

warnings.filterwarnings("ignore")
setup_logging("WARNING")

MODELS = ["lightgbm", "xgboost"]
SUFFIX = "_no_market"


def date_constant_columns(X: pd.DataFrame) -> list[str]:
    """Columns with one value per date: the market state, whatever it is called."""
    spread = X.groupby(level="date").std().max()
    return sorted(spread[spread.fillna(0) < 1e-9].index)


def main() -> None:
    t0 = time.time()
    panel = experiment.load_sharadar_panel(horizons=lc.HORIZONS, seq_len=lc.SEQ_LEN)
    print(f"Panel: {len(panel):,} rows ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for h in lc.HORIZONS:
        X, y, _ = split_xy(panel, h)
        market = date_constant_columns(X)
        print(f"h={h}: dropping {len(market)} market columns: {market}", flush=True)
        X = X.drop(columns=market)
        dates = X.index.get_level_values("date")
        for fold in lc.FOLD_ENDS:
            window = (dates <= pd.Timestamp(fold)) & dates.isin(dates.unique().sort_values()[::lc.TRAIN_STRIDE])
            Xw, yw = X[window], y[window]
            tr, va = _time_split(Xw, 0.1, h)
            X_tr, y_tr, X_va, y_va = Xw[tr], yw[tr], Xw[va], yw[va]
            probe = np.sort(np.random.default_rng(0).choice(len(X_tr), min(len(X_tr), lc.PROBE_ROWS),
                                                            replace=False))
            for name in MODELS:
                m = models.build(name).fit_with_curve(X_tr, y_tr, X_va, y_va, X_tr.iloc[probe], y_tr.iloc[probe])
                n = len(m.history_["valid"]["logloss"])
                rows += lc._rows(fold, h, name + SUFFIX, "tree", range(1, n + 1), m.history_)
                print(f"  {fold} h={h} {name}: done", flush=True)

    new = pd.DataFrame(rows)
    path = lc.OUT_DIR / "curves.parquet"
    curves = pd.read_parquet(path)
    curves = pd.concat([curves[~curves["model"].str.endswith(SUFFIX)], new], ignore_index=True)
    curves.to_parquet(path, index=False)
    curves.to_parquet(lc.EXPORT_FILE, index=False)

    # Final training and validation loss, with and without the market columns.
    last = curves[(curves["metric"] == "logloss") & curves["model"].str.startswith(tuple(MODELS))]
    last = last.sort_values("step").groupby(["model", "horizon", "fold_end", "split"])["value"].last()
    best = curves[(curves["metric"] == "logloss") & (curves["split"] == "valid")
                  & curves["model"].str.startswith(tuple(MODELS))]
    best = best.groupby(["model", "horizon", "fold_end"])["value"].min().rename("valid_best")
    table = last.unstack("split").join(best)
    table.to_csv(lc.OUT_DIR / "market_ablation.csv")
    print(table.round(4).to_string())
    print(f"\nTotal {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
