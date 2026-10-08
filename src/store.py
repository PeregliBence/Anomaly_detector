"""Research store: experiments and their out-of-sample predictions in SQLite.

Every walk-forward run is recorded with what is needed to reproduce and audit
it — model, horizon, target, universe, protocol and the git commit of the code
that produced it — next to every prediction it made. The thesis tables and the
web interface read from here, so a number in either can always be traced back
to the run that produced it.

The database lives under data/, outside version control: its contents derive
from the licensed Sharadar snapshot.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT, STORE_PATH

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    name          TEXT NOT NULL,
    model         TEXT NOT NULL,
    horizon       INTEGER NOT NULL,
    target        TEXT NOT NULL,
    universe      TEXT NOT NULL,
    protocol      TEXT NOT NULL,
    git_commit    TEXT,
    metrics       TEXT
);
CREATE INDEX IF NOT EXISTS ix_experiments_lookup
    ON experiments (name, model, horizon, target, universe);

CREATE TABLE IF NOT EXISTS predictions (
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    date          TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    score         REAL NOT NULL,
    y_true        REAL,
    fwd_return    REAL,
    PRIMARY KEY (experiment_id, date, ticker)
) WITHOUT ROWID;

-- Per-date information coefficient, precomputed so charts need not read
-- millions of prediction rows.
CREATE TABLE IF NOT EXISTS daily_ic (
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    date          TEXT NOT NULL,
    ic            REAL NOT NULL,
    PRIMARY KEY (experiment_id, date)
) WITHOUT ROWID;
"""


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, capture_output=True, text=True, timeout=10)
        commit = out.stdout.strip()
        return f"{commit}-dirty" if commit and dirty.stdout.strip() else (commit or None)
    except Exception:
        return None


@contextmanager
def connect(path: Path | None = None):
    path = path or STORE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _jsonable(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, float)):
            out[k] = None if not np.isfinite(v) else float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = v if isinstance(v, (str, bool, type(None))) else str(v)
    return out


def save_experiment(
    name: str,
    model: str,
    horizon: int,
    target: str,
    universe: str,
    protocol: dict,
    predictions: pd.DataFrame,
    metrics: dict | None = None,
    replace: bool = True,
) -> int:
    """Record one run and its predictions. Returns the experiment id.

    With `replace`, an earlier run with the same name, model, horizon, target
    and universe is removed first, so re-running a backfill does not leave
    stale duplicates beside the fresh result.
    """
    with connect() as con:
        if replace:
            con.execute(
                "DELETE FROM experiments WHERE name=? AND model=? AND horizon=? "
                "AND target=? AND universe=?",
                (name, model, horizon, target, universe),
            )

        cur = con.execute(
            "INSERT INTO experiments (created_at, name, model, horizon, target, universe, "
            "protocol, git_commit, metrics) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                name, model, horizon, target, universe,
                json.dumps(protocol, sort_keys=True),
                _git_commit(),
                json.dumps(_jsonable(metrics or {})),
            ),
        )
        exp_id = cur.lastrowid

        rows = predictions.reset_index()[["date", "ticker", "score", "y_true", "fwd_return"]]
        rows = rows.assign(
            experiment_id=exp_id,
            date=pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d"),
        )
        rows = rows.astype({"score": float, "y_true": float, "fwd_return": float})
        rows = rows.where(pd.notna(rows), None)
        con.executemany(
            "INSERT INTO predictions (experiment_id, date, ticker, score, y_true, fwd_return) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows[["experiment_id", "date", "ticker", "score", "y_true", "fwd_return"]]
            .itertuples(index=False, name=None),
        )

        from src.metrics import rank_ic
        ic = rank_ic(predictions)
        con.executemany(
            "INSERT INTO daily_ic (experiment_id, date, ic) VALUES (?, ?, ?)",
            ((exp_id, d.strftime("%Y-%m-%d"), float(v)) for d, v in ic.items()),
        )

    logger.info("Stored experiment %d: %s/%s h=%d (%d predictions)",
                exp_id, name, model, horizon, len(rows))
    return exp_id


def experiments() -> pd.DataFrame:
    """All recorded runs with their metrics expanded into columns."""
    with connect() as con:
        df = pd.read_sql("SELECT * FROM experiments ORDER BY id", con)
    if df.empty:
        return df
    metrics = pd.json_normalize(df["metrics"].map(json.loads))
    return pd.concat([df.drop(columns=["metrics"]), metrics], axis=1)


def predictions(experiment_id: int) -> pd.DataFrame:
    """One run's predictions, indexed by (date, ticker) like the walk-forward output."""
    with connect() as con:
        df = pd.read_sql(
            "SELECT date, ticker, score, y_true, fwd_return FROM predictions "
            "WHERE experiment_id = ? ORDER BY date, ticker",
            con, params=(experiment_id,),
        )
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "ticker"])


def daily_ic(experiment_id: int) -> pd.Series:
    """One run's per-date information coefficient."""
    with connect() as con:
        df = pd.read_sql("SELECT date, ic FROM daily_ic WHERE experiment_id = ? ORDER BY date",
                         con, params=(experiment_id,))
    return pd.Series(df["ic"].values, index=pd.to_datetime(df["date"]), name="ic")


def find(name: str, model: str, horizon: int, target: str = "cross_sectional",
         universe: str = "point_in_time") -> int | None:
    """Id of the latest run matching the key, or None."""
    with connect() as con:
        row = con.execute(
            "SELECT id FROM experiments WHERE name=? AND model=? AND horizon=? "
            "AND target=? AND universe=? ORDER BY id DESC LIMIT 1",
            (name, model, horizon, target, universe),
        ).fetchone()
    return row[0] if row else None
