"""Export research summaries that can be committed and published.

The research store holds every prediction and derives from the licensed
Sharadar snapshot, so it stays local. Its summaries do not: per-experiment
metrics, the daily information coefficient and the report tables are
statistics about the models, not the vendor's data. This writes those to
research/, which the web interface reads when the store is not present —
for instance on a hosted deployment.
"""

from __future__ import annotations

import logging
import shutil

import pandas as pd

from src import store
from src.config import RESULTS_DIR, ROOT

logger = logging.getLogger(__name__)

EXPORT_DIR = ROOT / "research"
NAMES = ("final", "final_alt", "tuned", "final_abs")
METRIC_COLUMNS = ["id", "name", "model", "horizon", "target", "universe", "ic_mean",
                  "ic_tstat_folds", "ic_positive_rate", "spread_mean", "auc", "accuracy",
                  "git_commit", "created_at"]
TABLES = {
    RESULTS_DIR / "report" / "models.csv": "portfolio.csv",
    RESULTS_DIR / "report" / "ensemble_ic_by_year.csv": "ensemble_ic_by_year.csv",
    RESULTS_DIR / "survivorship" / "cross_sectional_by_universe.csv": "survivorship.csv",
    RESULTS_DIR / "survivorship" / "absolute_direction.csv": "absolute_direction.csv",
    RESULTS_DIR / "final" / "absolute_summary.csv": "absolute_summary.csv",
    RESULTS_DIR / "final" / "absolute_three_way.csv": "absolute_three_way.csv",
}


def export(names: tuple[str, ...] = NAMES) -> list[str]:
    EXPORT_DIR.mkdir(exist_ok=True)
    experiments = store.experiments()
    experiments = experiments[experiments["name"].isin(names)]
    if experiments.empty:
        raise RuntimeError(f"No experiments named {names} in the research store.")

    experiments[[c for c in METRIC_COLUMNS if c in experiments]].to_csv(
        EXPORT_DIR / "experiments.csv", index=False)

    ic = pd.concat([store.daily_ic(int(i)).rename("ic").to_frame().assign(experiment_id=int(i))
                    for i in experiments["id"]])
    ic.index.name = "date"
    ic.reset_index().to_parquet(EXPORT_DIR / "daily_ic.parquet", index=False)

    written = ["experiments.csv", "daily_ic.parquet"]
    for source, target in TABLES.items():
        if source.exists():
            shutil.copyfile(source, EXPORT_DIR / target)
            written.append(target)
        else:
            logger.warning("Not exported, missing: %s", source)
    return written
