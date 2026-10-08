"""Two comparisons the thesis needs.

1. Survivorship bias, measured. The same pipeline runs on three universes:
   point-in-time (correct), survivors (later-delisted companies removed) and
   today's list (today's liquid names applied to all history, as a study built
   on a current screener would). The gap between them is the bias.

2. Absolute vs relative direction. The task asks for the expected direction
   of the price. The absolute target ("will it rise?") is run on the correct
   universe next to the cross-sectional one ("will it beat its peers?"), so
   the two framings can be compared on the same data and protocol.

The point-in-time cross-sectional results come from the horizon sweep, which
used the identical protocol, instead of being recomputed.
"""

from __future__ import annotations

import gc
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import experiment
from src.config import setup_logging

warnings.filterwarnings("ignore")
setup_logging("WARNING")

HORIZONS = [5, 20]
CS_MODELS = ["random", "cs_momentum", "cs_reversal", "cs_low_vol", "logistic", "lightgbm"]
ABS_MODELS = ["random", "constant", "logistic", "lightgbm"]
PROTOCOL = dict(refit_every=63, train_stride=5)
OUT_DIR = Path("results/survivorship")


def names_per_date(panel: pd.DataFrame) -> float:
    return float(panel.groupby(level="date").size().mean())


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sweep = pd.read_csv("results/horizon_sweep/scorecard.csv")
    sweep = sweep[sweep["horizon"].isin(HORIZONS)].assign(universe="point_in_time")
    frames = [sweep]

    for mode in ("survivors", "todays_list"):
        t = time.time()
        panel = experiment.load_sharadar_panel(horizons=HORIZONS, universe_mode=mode)
        print(f"\n### {mode}: {len(panel):,} rows, "
              f"{names_per_date(panel):.0f} names per date on average", flush=True)
        for h in HORIZONS:
            _, sc = experiment.run(panel, horizon=h, model_names=CS_MODELS, **PROTOCOL)
            frames.append(sc.assign(universe=mode))
        print(f"  done in {(time.time() - t) / 60:.0f} min", flush=True)
        del panel
        gc.collect()

    cs = pd.concat(frames, ignore_index=True)
    cs.to_csv(OUT_DIR / "cross_sectional_by_universe.csv", index=False)

    print("\n===== Cross-sectional IC by universe =====")
    for h in HORIZONS:
        pivot = cs[cs["horizon"] == h].pivot(index="model", columns="universe", values="ic_mean")
        print(f"\nh={h}")
        print(pivot[["point_in_time", "survivors", "todays_list"]].round(4).to_string())

    print("\n===== Long-short decile spread by universe (mean per period) =====")
    for h in HORIZONS:
        pivot = cs[cs["horizon"] == h].pivot(index="model", columns="universe", values="spread_mean")
        print(f"\nh={h}")
        print(pivot[["point_in_time", "survivors", "todays_list"]].round(4).to_string())

    t = time.time()
    panel = experiment.load_sharadar_panel(horizons=HORIZONS)
    rows = []
    for h in HORIZONS:
        _, sc = experiment.run(panel, horizon=h, model_names=ABS_MODELS,
                               target_mode="absolute", **PROTOCOL)
        rows.append(sc)
    absolute = pd.concat(rows, ignore_index=True)
    absolute.to_csv(OUT_DIR / "absolute_direction.csv", index=False)

    print(f"\n===== Absolute direction, point-in-time universe ({(time.time() - t) / 60:.0f} min) =====")
    cols = ["horizon", "model", "accuracy", "majority_baseline", "edge_vs_majority",
            "auc", "brier", "ic_mean"]
    print(absolute[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
