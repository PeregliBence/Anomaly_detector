"""Learning curves: do the models train properly, and are the settings sensible?

For three training windows (ending 2012, 2018 and 2025) and both horizons,
every learned model is fitted exactly as in the walk-forward — same columns,
same stride, same hyperparameters — while recording the loss on the training
data and on a purged validation slice: the last 10% of the window's dates,
separated from the training rows by `horizon` dates so overlapping labels do
not leak into it.

- LightGBM, XGBoost: log loss and AUC after every tree.
- MLP, LSTM: after every epoch, per seed (their own early-stopping split).
- Logistic: along the regularisation path, since it has no iterations.

The reference line is the log loss of predicting the validation base rate for
everyone. A model below it has learned something; the gap between the train
and validation curves shows over- or underfitting.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import experiment, models
from src.config import ROOT, setup_logging
from src.neural import _time_split
from src.panel import split_xy

warnings.filterwarnings("ignore")
setup_logging("WARNING")

HORIZONS = [5, 20]
FOLD_ENDS = ["2012-12-31", "2018-12-31", "2025-12-31"]
MODELS = ["logistic", "lightgbm", "xgboost", "mlp", "lstm"]
LOGISTIC_C = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0]
TRAIN_STRIDE = 5
SEQ_LEN = 20
PROBE_ROWS = 100_000
OUT_DIR = ROOT / "results" / "learning_curves"
EXPORT_FILE = ROOT / "research" / "learning_curves.parquet"


def _logloss(y, p) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _rows(fold, h, model, step_name, steps, history, seed=0):
    """Long-format rows from {split: {metric: [values per step]}}."""
    out = []
    for split, metrics in history.items():
        for metric, values in metrics.items():
            out += [dict(fold_end=fold, horizon=h, model=model, seed=seed, step_name=step_name,
                         step=s, split=split, metric=metric, value=float(v))
                    for s, v in zip(steps, values)]
    return out


def _logistic_path(X_tr, y_tr, X_va, y_va, X_pr, y_pr):
    from sklearn.metrics import roc_auc_score

    hist = {"train": {"logloss": [], "auc": []}, "valid": {"logloss": [], "auc": []}}
    for C in LOGISTIC_C:
        m = models.Logistic(C=C).fit(X_tr, y_tr)
        for split, X_, y_ in (("train", X_pr, y_pr), ("valid", X_va, y_va)):
            p = m.predict_proba(X_)
            hist[split]["logloss"].append(_logloss(y_.values, p))
            hist[split]["auc"].append(roc_auc_score(y_.values, p))
    return hist


def curves_for_window(X, y, fold, h) -> list[dict]:
    dates = X.index.get_level_values("date")
    window = (dates <= pd.Timestamp(fold)) & dates.isin(dates.unique().sort_values()[::TRAIN_STRIDE])
    Xw, yw = X[window], y[window]

    train_mask, val_mask = _time_split(Xw, 0.1, h)
    X_tr, y_tr, X_va, y_va = Xw[train_mask], yw[train_mask], Xw[val_mask], yw[val_mask]
    probe = np.sort(np.random.default_rng(0).choice(len(X_tr), min(len(X_tr), PROBE_ROWS), replace=False))
    X_pr, y_pr = X_tr.iloc[probe], y_tr.iloc[probe]

    rows = [dict(fold_end=fold, horizon=h, model="reference", seed=0, step_name="", step=0,
                 split="valid", metric="logloss", value=_logloss(y_va.values, y_tr.mean()))]
    for name in MODELS:
        t = time.time()
        if name == "logistic":
            rows += _rows(fold, h, name, "C", LOGISTIC_C, _logistic_path(X_tr, y_tr, X_va, y_va, X_pr, y_pr))
        elif name in ("lightgbm", "xgboost"):
            m = models.build(name).fit_with_curve(X_tr, y_tr, X_va, y_va, X_pr, y_pr)
            n = len(m.history_["valid"]["logloss"])
            rows += _rows(fold, h, name, "tree", range(1, n + 1), m.history_)
        else:
            m = models.build(name, horizon=h).fit(Xw, yw)
            for c in m.history_:
                hist = {"train": {"logloss": c["train_loss"]},
                        "valid": {"logloss": c["val_loss"], "auc": c["val_auc"]}}
                rows += _rows(fold, h, name, "epoch", range(1, len(c["val_loss"]) + 1), hist, seed=c["seed"])
                rows.append(dict(fold_end=fold, horizon=h, model=name, seed=c["seed"], step_name="epoch",
                                 step=c["best_epoch"], split="valid", metric="best_epoch", value=c["best_epoch"]))
        print(f"  {fold} h={h} {name}: {time.time() - t:.0f}s", flush=True)
    return rows


def summarise(curves: pd.DataFrame) -> pd.DataFrame:
    ref = curves[curves["model"] == "reference"].set_index(["fold_end", "horizon"])["value"]
    out = []
    loss = curves[(curves["metric"] == "logloss") & (curves["model"] != "reference")]
    for (fold, h, model, seed), g in loss.groupby(["fold_end", "horizon", "model", "seed"]):
        tr = g[g["split"] == "train"].set_index("step")["value"]
        va = g[g["split"] == "valid"].set_index("step")["value"]
        auc = curves[(curves["fold_end"] == fold) & (curves["horizon"] == h) & (curves["model"] == model)
                     & (curves["seed"] == seed) & (curves["metric"] == "auc") & (curves["split"] == "valid")]
        auc = auc.set_index("step")["value"]
        used = va.index[-1]  # the setting the walk-forward actually uses
        if model == "logistic":
            used = models.Logistic().C
        elif model in ("mlp", "lstm"):
            used = va.idxmin()  # early stopping restores the best epoch
        out.append(dict(fold_end=fold, horizon=h, model=model, seed=seed,
                        reference_logloss=ref[(fold, h)],
                        used_step=used, best_step=va.idxmin(), last_step=va.index[-1],
                        valid_logloss_used=va[used], valid_logloss_best=va.min(),
                        train_logloss_used=tr[used], valid_auc_used=auc.get(used, np.nan)))
    s = pd.DataFrame(out)
    s["gain_vs_reference"] = s["reference_logloss"] - s["valid_logloss_used"]
    s["train_valid_gap"] = s["valid_logloss_used"] - s["train_logloss_used"]
    return s


def plot(curves: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = curves[curves["model"] == "reference"].set_index(["fold_end", "horizon"])["value"]
    for model in MODELS:
        d = curves[(curves["model"] == model) & (curves["metric"] == "logloss")]
        fig, axes = plt.subplots(len(HORIZONS), len(FOLD_ENDS), figsize=(13, 6.5), squeeze=False)
        for i, h in enumerate(HORIZONS):
            for j, fold in enumerate(FOLD_ENDS):
                ax = axes[i, j]
                for (split, seed), g in d[(d["horizon"] == h) & (d["fold_end"] == fold)].groupby(["split", "seed"]):
                    ax.plot(g["step"], g["value"], color="tab:blue" if split == "train" else "tab:orange",
                            alpha=0.8, label=split if seed == 0 else None)
                ax.axhline(ref[(fold, h)], color="grey", ls="--", lw=1, label="base rate")
                if model == "logistic":
                    ax.set_xscale("log")
                ax.set_title(f"h={h}, train to {fold[:4]}", fontsize=9)
                ax.tick_params(labelsize=8)
        step = d["step_name"].iloc[0] if len(d) else ""
        fig.supxlabel(step)
        fig.supylabel("log loss")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{model}: training vs purged validation loss")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{model}.png", dpi=120)
        plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    panel = experiment.load_sharadar_panel(horizons=HORIZONS, seq_len=SEQ_LEN)
    print(f"Panel: {len(panel):,} rows ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for h in HORIZONS:
        X, y, _ = split_xy(panel, h)
        for fold in FOLD_ENDS:
            rows += curves_for_window(X, y, fold, h)
            pd.DataFrame(rows).to_parquet(OUT_DIR / "curves.parquet", index=False)

    curves = pd.DataFrame(rows)
    summary = summarise(curves)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    curves.to_parquet(EXPORT_FILE, index=False)
    plot(curves)

    pd.set_option("display.width", 220)
    print(summary.drop(columns=["seed"]).groupby(["model", "horizon", "fold_end"]).mean().round(4).to_string())
    print(f"\nTotal {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
