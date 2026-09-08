"""
Compare two forecast benchmark outputs and generate thesis-ready summary tables.

Usage:
  python compare_forecast_runs.py \
    --baseline-dir results/thesis \
    --candidate-dir results/thesis_ml \
    --label-baseline baseline \
    --label-candidate ml \
    --out-dir results/comparisons/latest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "ticker",
    "models",
    "directional_accuracy",
    "sharpe_ratio",
    "max_drawdown_pct",
    "profit_factor",
    "historical_paper_return_pct",
]

OPTIONAL_COLS = [
    "trade_count",
    "trade_rate",
    "win_rate",
    "projected_return_pct",
    "round_trip_cost_bps",
    "min_confidence",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two forecast benchmark runs")
    parser.add_argument("--baseline-dir", required=True, help="Folder containing forecast_benchmark.csv")
    parser.add_argument("--candidate-dir", required=True, help="Folder containing forecast_benchmark.csv")
    parser.add_argument("--label-baseline", default="baseline", help="Display label for baseline")
    parser.add_argument("--label-candidate", default="candidate", help="Display label for candidate")
    parser.add_argument("--out-dir", default="results/comparisons/latest", help="Output folder")
    return parser.parse_args()


def _load_forecast_csv(folder: str) -> pd.DataFrame:
    path = Path(folder) / "forecast_benchmark.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    for c in REQUIRED_COLS + OPTIONAL_COLS:
        if c not in df.columns:
            df[c] = np.nan

    return df


def _best_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    if "row_type" in df.columns:
        df = df[df["row_type"].fillna("model") == "model"].copy()
    sort_cols = ["ticker", "directional_accuracy", "sharpe_ratio", "historical_paper_return_pct"]
    best = (
        df.sort_values(sort_cols, ascending=[True, False, False, False])
        .groupby("ticker", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best


def _summary_row(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "label": label,
            "rows": 0,
            "tickers": 0,
            "avg_directional_accuracy": np.nan,
            "avg_sharpe": np.nan,
            "avg_max_drawdown_pct": np.nan,
            "avg_profit_factor": np.nan,
            "avg_historical_paper_return_pct": np.nan,
            "avg_projected_return_pct": np.nan,
            "avg_trade_count": np.nan,
            "avg_trade_rate": np.nan,
            "avg_win_rate": np.nan,
        }

    return {
        "label": label,
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
        "avg_directional_accuracy": float(df["directional_accuracy"].mean()),
        "avg_sharpe": float(df["sharpe_ratio"].mean()),
        "avg_max_drawdown_pct": float(df["max_drawdown_pct"].mean()),
        "avg_profit_factor": float(df["profit_factor"].mean()),
        "avg_historical_paper_return_pct": float(df["historical_paper_return_pct"].mean()),
        "avg_projected_return_pct": float(df["projected_return_pct"].mean()),
        "avg_trade_count": float(df["trade_count"].mean()),
        "avg_trade_rate": float(df["trade_rate"].mean()),
        "avg_win_rate": float(df["win_rate"].mean()),
    }


def _fmt(v, pct: bool = False) -> str:
    if pd.isna(v):
        return "n/a"
    if pct:
        return f"{v * 100:.2f}%"
    return f"{v:.4f}"


def _build_markdown_report(
    baseline_all: pd.DataFrame,
    candidate_all: pd.DataFrame,
    baseline_best: pd.DataFrame,
    candidate_best: pd.DataFrame,
    baseline_label: str,
    candidate_label: str,
) -> str:
    sum_all = pd.DataFrame([
        _summary_row(baseline_all, baseline_label),
        _summary_row(candidate_all, candidate_label),
    ])

    merged_best = baseline_best.merge(
        candidate_best,
        on="ticker",
        suffixes=(f"_{baseline_label}", f"_{candidate_label}"),
        how="inner",
    )

    lines = []
    lines.append("# Forecast Run Comparison")
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Label | Rows | Tickers | Avg DirAcc | Avg Sharpe | Avg MaxDD% | Avg PF | Avg Hist Return% | Avg Proj Return% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in sum_all.iterrows():
        lines.append(
            "| {label} | {rows} | {tickers} | {diracc} | {sharpe} | {dd} | {pf} | {hist} | {proj} |".format(
                label=r["label"],
                rows=int(r["rows"]),
                tickers=int(r["tickers"]),
                diracc=_fmt(r["avg_directional_accuracy"], pct=True),
                sharpe=_fmt(r["avg_sharpe"]),
                dd=_fmt(r["avg_max_drawdown_pct"]),
                pf=_fmt(r["avg_profit_factor"]),
                hist=_fmt(r["avg_historical_paper_return_pct"]),
                proj=_fmt(r["avg_projected_return_pct"]),
            )
        )

    lines.append("")
    lines.append("## Best-Per-Ticker Delta")
    lines.append("")

    if merged_best.empty:
        lines.append("No overlapping tickers between the two runs.")
        return "\n".join(lines)

    lines.append("| Ticker | DirAcc Δ (cand-base) | Sharpe Δ | HistReturn% Δ | Candidate Models |")
    lines.append("|---|---:|---:|---:|---|")

    for _, r in merged_best.iterrows():
        d_dir = r[f"directional_accuracy_{candidate_label}"] - r[f"directional_accuracy_{baseline_label}"]
        d_sh = r[f"sharpe_ratio_{candidate_label}"] - r[f"sharpe_ratio_{baseline_label}"]
        d_ret = r[f"historical_paper_return_pct_{candidate_label}"] - r[f"historical_paper_return_pct_{baseline_label}"]
        lines.append(
            f"| {r['ticker']} | {d_dir*100:.2f}% | {d_sh:.4f} | {d_ret:.2f} | {r[f'models_{candidate_label}']} |"
        )

    return "\n".join(lines)


def main() -> None:
    args = _parse_args()

    baseline_all = _load_forecast_csv(args.baseline_dir)
    candidate_all = _load_forecast_csv(args.candidate_dir)

    baseline_best = _best_per_ticker(baseline_all)
    candidate_best = _best_per_ticker(candidate_all)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_best = baseline_best.merge(
        candidate_best,
        on="ticker",
        suffixes=(f"_{args.label_baseline}", f"_{args.label_candidate}"),
        how="outer",
    )

    report = _build_markdown_report(
        baseline_all,
        candidate_all,
        baseline_best,
        candidate_best,
        args.label_baseline,
        args.label_candidate,
    )

    (out_dir / "comparison_report.md").write_text(report, encoding="utf-8")
    baseline_best.to_csv(out_dir / f"best_per_ticker_{args.label_baseline}.csv", index=False)
    candidate_best.to_csv(out_dir / f"best_per_ticker_{args.label_candidate}.csv", index=False)
    merged_best.to_csv(out_dir / "best_per_ticker_merged.csv", index=False)

    print(f"Comparison saved to: {out_dir}")


if __name__ == "__main__":
    main()
