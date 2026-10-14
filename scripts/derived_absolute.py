"""A derived absolute-direction forecast: market base rate plus relative rank.

The direct absolute models try to time the market and fail: a fifth of the
label's variance is the day's market-wide move, learnable from only a few
hundred independent periods, and they end up fitting past episodes. This
forecast refuses to time the market. Its probability of a rise is the
historical share of rises, shifted by the stock's within-day rank from the
cross-sectional ensemble:

    P(up) = sigmoid(a + b * (rank - 0.5))

Each year, b is fitted by logistic regression on earlier years' out-of-sample
ranks and realised directions, and a is set to the log-odds of the long-run
share of rises in the whole panel since 2005 — both from outcomes resolved
before the year starts. A first version re-estimated a from the out-of-sample
years alone; in 2009 that meant 2008 alone, and the method called the 2009
rally a fall: market timing again, by the back door. Because the rank is
uniform within every day and a moves only slowly, there are no market calls.

All three approaches (always up, the direct absolute ensemble, this one) are
scored on the same dates. Coefficients fitted on all history are saved for
the live run.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import store
from src.config import setup_logging

warnings.filterwarnings("ignore")
setup_logging("WARNING")

ENSEMBLE = ["lightgbm", "xgboost", "mlp", "lstm"]
MODELS_DIR = Path("models")


def ensemble(name: str, target: str, horizon: int, ranks: bool) -> pd.DataFrame:
    parts = [store.predictions(store.find(name, m, horizon, target)) for m in ENSEMBLE]
    scores = [p["score"].groupby(level="date").rank(pct=True) if ranks else p["score"] for p in parts]
    return parts[0][["fwd_return"]].assign(score=pd.concat(scores, axis=1).mean(axis=1))


def scores(p: np.ndarray, y: np.ndarray, dates: pd.Index) -> dict:
    up = pd.Series(p > 0.5, index=dates).groupby(level=0).mean()
    return {"accuracy": float(((p > 0.5) == y).mean()), "always_up_accuracy": float(y.mean()),
            "edge_vs_always_up": float(((p > 0.5) == y).mean() - y.mean()),
            "predicted_up_share": float((p > 0.5).mean()), "daily_up_share_sd": float(up.std()),
            "auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "brier_base_rate": float(brier_score_loss(y, np.full(len(y), y.mean())))}


def fit(rank: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=1e6).fit((rank - 0.5).reshape(-1, 1), y)


def within_day_auc(p: pd.Series, y: pd.Series) -> float:
    joined = pd.DataFrame({"p": p, "y": y})
    per_day = joined.groupby(level="date").apply(
        lambda g: roc_auc_score(g["y"], g["p"]) if g["y"].nunique() == 2 and g["p"].nunique() > 1
        else np.nan)
    return float(per_day.mean())


def long_run_rises(horizons) -> dict[int, pd.Series]:
    """Daily share of rises in the full panel since 2005, per horizon."""
    from src import experiment
    panel = experiment.load_sharadar_panel(horizons=list(horizons))
    return {h: (panel[f"fwd_return_{h}"].dropna() > 0).groupby(level="date").agg(["sum", "count"])
            for h in horizons}


def main() -> None:
    pd.set_option("display.width", 200)
    rows, coefficients = [], {}
    rises = long_run_rises((5, 20))

    for h in (5, 20):
        rel = ensemble("final", "cross_sectional", h, ranks=True).dropna()
        absolute = ensemble("final_abs", "absolute", h, ranks=False)["score"]
        rel["y"] = (rel["fwd_return"] > 0).astype(int)
        dates = rel.index.get_level_values("date")
        calendar = dates.unique().sort_values()

        out = []
        for year in sorted(set(dates.year))[1:]:
            test = rel[dates.year == year]
            first = test.index.get_level_values("date").min()
            cutoff = calendar[calendar.get_loc(first) - h]  # purge: outcomes resolved before the year
            train = rel[dates <= cutoff]
            slope = fit(train["score"].values, train["y"].values).coef_[0, 0]
            history = rises[h][rises[h].index <= cutoff].sum()
            intercept = np.log(history["sum"] / (history["count"] - history["sum"]))
            p = 1 / (1 + np.exp(-(intercept + slope * (test["score"].values - 0.5))))
            out.append(pd.Series(p, index=test.index))
        derived = pd.concat(out)

        common = derived.index.intersection(absolute.dropna().index)
        y = rel.loc[common, "y"].values
        d = common.get_level_values("date")
        for label, p in (("always_up", np.ones(len(common))),
                         ("absolute_ensemble", absolute.loc[common].values),
                         ("drift_plus_rank", derived.loc[common].values)):
            row = {"horizon": h, "approach": label, "test_start": str(d.min().date()), **scores(p, y, d)}
            row["within_day_auc"] = within_day_auc(pd.Series(p, index=common), pd.Series(y, index=common))
            if label == "always_up":
                row.update(auc=float("nan"), brier=float("nan"), within_day_auc=float("nan"))
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

        preds = rel.loc[common, ["fwd_return"]].assign(score=derived.loc[common].values,
                                                       y_true=y.astype(float))
        store.save_experiment("final_abs", "drift_plus_rank", h, "absolute", "point_in_time",
                              {"method": "logistic on cross-sectional ensemble rank, refit yearly"},
                              preds, rows[-1])

        total = rises[h].sum()
        base_rate = float(total["sum"] / total["count"])
        coefficients[h] = {"intercept": float(np.log(base_rate / (1 - base_rate))),
                           "slope": float(fit(rel["score"].values, rel["y"].values).coef_[0, 0]),
                           "base_rate": base_rate,
                           "base_rate_from": str(rises[h].index.min().date()),
                           "slope_fitted_on": f"{dates.min().date()}..{dates.max().date()}"}

    table = pd.DataFrame(rows)
    table.to_csv("results/final/absolute_three_way.csv", index=False)
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "drift_plus_rank.json").write_text(json.dumps(coefficients, indent=2))
    print("\n" + table.round(4).to_string(index=False))
    print("\nCoefficients for live use:", json.dumps(coefficients, indent=2))


if __name__ == "__main__":
    main()
