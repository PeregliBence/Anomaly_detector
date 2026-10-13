"""Thesis tables, built from the research store in one pass.

    python -m src.cli report

Writes CSV tables and a Markdown summary to results/report/. Every number is
computed from predictions in the store, so each traces back to a recorded
run with its protocol and commit.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import portfolio, store
from src.config import RESULTS_DIR
from src.metrics import newey_west_tstat, rank_ic

logger = logging.getLogger(__name__)

OUT_DIR = RESULTS_DIR / "report"
ENSEMBLE = ["lightgbm", "xgboost", "mlp", "lstm"]
VARIANTS = {"final": "Ár- és piaci jellemzők", "final_alt": "+ alternatív adat"}


def _ic_stats(preds: pd.DataFrame, horizon: int) -> dict:
    ic = rank_ic(preds)
    t, _ = newey_west_tstat(ic, lags=max(horizon - 1, 252))
    return {"ic_mean": ic.mean(), "ic_tstat": t, "ic_positive_rate": (ic > 0).mean()}


def _portfolio_stats(preds: pd.DataFrame, horizon: int) -> dict:
    s = portfolio.summarise(portfolio.long_short_returns(preds, horizon), horizon)
    offsets = portfolio.sharpe_over_offsets(preds, horizon)
    return {
        "ls_ann_return": s["long_short_ann_return"],
        "ls_ann_vol": s["long_short_ann_vol"],
        "ls_sharpe": s["long_short_sharpe"],
        "ls_sharpe_min": offsets.min(),
        "ls_sharpe_median": offsets.median(),
        "ls_sharpe_max": offsets.max(),
        "ls_max_drawdown": s["long_short_max_drawdown"],
        "ls_hit_rate": s["long_short_hit_rate"],
        "turnover": s.get("turnover_per_rebalance"),
        "ls_sharpe_net_5bps": s.get("long_short_sharpe_net_5bps"),
        "ls_sharpe_net_10bps": s.get("long_short_sharpe_net_10bps"),
        "ls_sharpe_net_20bps": s.get("long_short_sharpe_net_20bps"),
    }


def _ensemble(members: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ranks = [p["score"].groupby(level="date").rank(pct=True) for p in members.values()]
    base = next(iter(members.values()))[["fwd_return"]]
    return base.assign(score=pd.concat(ranks, axis=1).mean(axis=1))


def model_table(experiments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every model and the ensemble, per variant and horizon; plus ensemble IC by year."""
    rows, yearly = [], {}
    for (name, horizon), group in experiments.groupby(["name", "horizon"]):
        members = {}
        for _, exp in group.iterrows():
            preds = store.predictions(int(exp["id"]))
            rows.append({"variant": name, "horizon": horizon, "model": exp["model"],
                         **_ic_stats(preds, horizon), **_portfolio_stats(preds, horizon)})
            if exp["model"] in ENSEMBLE:
                members[exp["model"]] = preds[["score", "fwd_return"]]
            logger.info("Reported %s %s h=%d", name, exp["model"], horizon)

        if set(ENSEMBLE) <= set(members):
            ens = _ensemble(members)
            rows.append({"variant": name, "horizon": horizon, "model": "ensemble",
                         **_ic_stats(ens, horizon), **_portfolio_stats(ens, horizon)})
            ic = rank_ic(ens)
            yearly[(name, horizon)] = ic.groupby(ic.index.year).mean()

    table = pd.DataFrame(rows)
    years = pd.DataFrame(yearly)
    years.columns = [f"{VARIANTS.get(n, n)}, {h} nap" for n, h in years.columns]
    return table, years


def _md_table(df: pd.DataFrame, index: bool = False) -> str:
    """A Markdown table without the optional `tabulate` dependency."""
    if index:
        df = df.reset_index()

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.3f}"
        return str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body])


BASE_COLUMNS = {
    "model": "Modell", "ic_mean": "IC", "ic_tstat": "t", "ic_positive_rate": "IC>0",
    "ls_ann_return": "LS éves hozam", "ls_sharpe": "Sharpe", "ls_sharpe_min": "Sharpe min",
    "ls_sharpe_max": "Sharpe max", "ls_max_drawdown": "Max. visszaesés",
}
COST_COLUMNS = {
    "turnover": "Forgás", "ls_sharpe_net_5bps": "Sharpe 5 bp",
    "ls_sharpe_net_10bps": "Sharpe 10 bp", "ls_sharpe_net_20bps": "Sharpe 20 bp",
}


def _markdown(table: pd.DataFrame, years: pd.DataFrame, costs: bool = False) -> str:
    intro = ("Teszt: 2008–2026, a legforgalmasabb 1000 amerikai részvény, túlélési torzítás nélkül; "
             "purged walk-forward, éves újratanítás. A long-short portfólió a felső és alsó decilis, "
             "egyenlő súllyal, a horizonttal megegyező, nem átfedő tartással, tranzakciós költségek "
             "nélkül (bruttó).")
    if costs:
        intro += (" A költségoszlopok egyirányú, arányos költséggel számolnak (bp = bázispont); "
                  "a forgás az újrasúlyozásonként lecserélt nevek aránya lábanként.")
    lines = ["# Végső kiértékelés", "", intro, ""]

    columns = {**BASE_COLUMNS, **(COST_COLUMNS if costs else {})}
    for h, group in table.groupby("horizon"):
        lines += [f"## {h} napos horizont", ""]
        view = group.assign(Változat=group["variant"].map(VARIANTS))
        view = view[["Változat", *columns]].rename(columns=columns)
        lines += [_md_table(view), ""]

    if not years.empty:
        lines += ["## Az ensemble IC-je évenként", "", _md_table(years, index=True), ""]
    return "\n".join(lines)


def build(names: tuple[str, ...] = ("final", "final_alt"), costs: bool = False) -> Path:
    """Write the tables. Gross results by default; `costs` adds the net-of-cost columns.

    The CSV always carries every column, so the cost analysis needs no rerun.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    experiments = store.experiments()
    experiments = experiments[experiments["name"].isin(names)]
    if experiments.empty:
        raise RuntimeError(f"No experiments named {names} in the store. Run the final evaluation first.")

    table, years = model_table(experiments)
    table.to_csv(OUT_DIR / "models.csv", index=False)
    years.to_csv(OUT_DIR / "ensemble_ic_by_year.csv")

    if {"final", "final_alt"} <= set(table["variant"]):
        gain = table.pivot_table(index=["horizon", "model"], columns="variant", values="ic_mean")
        gain["gain"] = gain["final_alt"] - gain["final"]
        gain.dropna().to_csv(OUT_DIR / "alternative_data_gain.csv")

    path = OUT_DIR / "report.md"
    path.write_text(_markdown(table, years, costs=costs), encoding="utf-8")
    return path


def render(costs: bool = False) -> Path:
    """Rewrite report.md from the saved CSVs, without recomputing anything."""
    table = pd.read_csv(OUT_DIR / "models.csv")
    years = pd.read_csv(OUT_DIR / "ensemble_ic_by_year.csv", index_col=0)
    path = OUT_DIR / "report.md"
    path.write_text(_markdown(table, years, costs=costs), encoding="utf-8")
    return path
