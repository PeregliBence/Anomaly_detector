"""
Simple forecast evaluator for thesis work.

What it does:
- trains forecast models on historical rolling windows
- predicts the next horizon (default: 5 trading days)
- compares predicted direction vs realized direction
- writes per-window and summary outputs

No paper-trading, no fees, no benchmark strategies, no GO/NO-GO gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.backtesting import walk_forward_forecast
from app.config import CONFIG, logger
from app.data_loader import load_data, validate_data
from app.features import add_all_features, prepare_ml_features
from app.forecasting import HAS_STATSMODELS, HAS_TF, HAS_XGBOOST, forecast_ensemble


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple weekly forecast evaluation")
    parser.add_argument("--tickers", default=None, help="Comma-separated Yahoo tickers")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD), default: today")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon in trading days")
    parser.add_argument(
        "--model-sets",
        default=None,
        help="Model sets split by ';' and model names split by ','",
    )
    parser.add_argument("--out-dir", default=None, help="Output folder")
    parser.add_argument("--profile", default="", help="Optional JSON profile file")
    parser.add_argument("--run-tag", default="", help="Optional run tag")
    return parser.parse_args()


def _parse_model_sets(text: str) -> list[list[str]]:
    model_sets = []
    for chunk in text.split(";"):
        names = [x.strip().lower() for x in chunk.split(",") if x.strip()]
        if names:
            model_sets.append(names)
    return model_sets


def _load_profile(path: str) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning("Profile file not found: %s", p)
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to parse profile file (%s): %s", p, e)
        return {}
    if not isinstance(data, dict):
        logger.warning("Profile must be a JSON object: %s", p)
        return {}
    return data


def _apply_profile(args: argparse.Namespace) -> argparse.Namespace:
    profile_path = args.profile.strip()
    if not profile_path:
        default_profile = Path("thesis_profile.json")
        if default_profile.exists():
            profile_path = str(default_profile)

    profile = _load_profile(profile_path)
    if not profile:
        return args

    args.profile = profile_path
    if args.tickers is None:
        args.tickers = str(profile.get("tickers", "AAPL,MSFT,NVDA,^GSPC,BTC-USD"))
    if args.start is None:
        args.start = str(profile.get("start", "2020-01-01"))
    if args.end is None:
        args.end = profile.get("end", None)
    if args.horizon is None:
        args.horizon = int(profile.get("horizon", 5))
    if args.out_dir is None:
        args.out_dir = str(profile.get("out_dir", "results/simple_forecast"))

    model_sets = profile.get("model_sets")
    if args.model_sets is None and isinstance(model_sets, list) and model_sets:
        model_chunks = []
        for group in model_sets:
            if isinstance(group, list):
                cleaned = [str(m).strip().lower() for m in group if str(m).strip()]
                if cleaned:
                    model_chunks.append(",".join(cleaned))
        if model_chunks:
            args.model_sets = ";".join(model_chunks)

    return args


def _prepare_forecast_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = add_all_features(raw_df, include_targets=True)
    _, _, _, _, ml_index = prepare_ml_features(df)
    if len(ml_index) == 0:
        return pd.DataFrame()
    return df.loc[ml_index].copy()


def _run_walk_forward(df: pd.DataFrame, ticker: str, models: list[str], horizon: int):
    availability = {
        "arima": HAS_STATSMODELS,
        "xgboost": HAS_XGBOOST,
        "lstm": HAS_TF,
    }
    active_models = [m for m in models if availability.get(m, False)]
    if not active_models:
        return None

    def forecast_fn(train_df: pd.DataFrame):
        result, _ = forecast_ensemble(train_df, horizon=horizon, models=active_models)
        return result

    bt = walk_forward_forecast(
        df,
        forecast_fn,
        window_size=CONFIG.backtest_window_size,
        step_size=horizon,
        horizon=horizon,
        min_train_size=200,
    )
    bt.ticker = ticker
    return bt


def _safe(v, nd=4):
    if v is None:
        return 0.0
    try:
        return round(float(v), nd)
    except Exception:
        return 0.0


def main() -> None:
    args = _parse_args()
    args = _apply_profile(args)

    if args.tickers is None:
        args.tickers = "AAPL,MSFT,NVDA,^GSPC,BTC-USD"
    if args.start is None:
        args.start = "2020-01-01"
    if args.horizon is None:
        args.horizon = 5
    if args.model_sets is None:
        args.model_sets = "xgboost;arima,xgboost;arima,xgboost,lstm"
    if args.out_dir is None:
        args.out_dir = "results/simple_forecast"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_tag = args.run_tag.strip() or pd.Timestamp.now().strftime("simple-%Y%m%d-%H%M%S")
    tickers = [x.strip() for x in args.tickers.split(",") if x.strip()]
    model_sets = _parse_model_sets(args.model_sets)

    summary_rows = []
    window_rows = []
    skipped = []

    logger.info("Starting simple forecast evaluator for %d tickers", len(tickers))

    for ticker in tickers:
        logger.info("Processing %s", ticker)
        raw = load_data(ticker, start=args.start, end=args.end)
        ok, warnings = validate_data(raw)

        if raw.empty:
            skipped.append({"ticker": ticker, "reason": "empty_data"})
            continue

        if warnings:
            for w in warnings:
                logger.warning("%s: %s", ticker, w)

        df = _prepare_forecast_frame(raw)
        min_rows = max(220, 200 + args.horizon * 2)
        if df.empty or len(df) < min_rows:
            skipped.append({"ticker": ticker, "reason": "insufficient_feature_rows"})
            continue

        for model_set in model_sets:
            bt = _run_walk_forward(df, ticker=ticker, models=model_set, horizon=args.horizon)
            if bt is None:
                skipped.append({"ticker": ticker, "reason": f"models_unavailable:{','.join(model_set)}"})
                continue

            summary_rows.append(
                {
                    "run_tag": run_tag,
                    "ticker": ticker,
                    "models": ",".join(model_set),
                    "n_windows": int(bt.n_windows),
                    "directional_accuracy": _safe(bt.directional_accuracy),
                    "mae": _safe(bt.mae),
                    "rmse": _safe(bt.rmse),
                }
            )

            for w in bt.window_results:
                window_rows.append(
                    {
                        "run_tag": run_tag,
                        "ticker": ticker,
                        "models": ",".join(model_set),
                        "train_end": w.get("train_end", ""),
                        "test_end": w.get("test_end", ""),
                        "predicted_dir": w.get("predicted_dir", ""),
                        "actual_dir": w.get("actual_dir", ""),
                        "correct": int(bool(w.get("correct", False))),
                        "confidence": _safe(w.get("confidence", 0.0), nd=4),
                        "actual_return_pct": _safe(w.get("actual_return_pct", 0.0), nd=4),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    windows_df = pd.DataFrame(window_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(["ticker", "directional_accuracy", "mae"], ascending=[True, False, True])

    summary_csv = out_dir / "forecast_summary.csv"
    windows_csv = out_dir / "forecast_windows.csv"
    run_details = out_dir / "run_details.json"
    report_md = out_dir / "simple_summary.md"

    summary_df.to_csv(summary_csv, index=False)
    windows_df.to_csv(windows_csv, index=False)

    details = {
        "run_tag": run_tag,
        "args": vars(args),
        "n_tickers": len(tickers),
        "n_summary_rows": int(summary_df.shape[0]),
        "n_window_rows": int(windows_df.shape[0]),
        "skipped": skipped,
    }
    run_details.write_text(json.dumps(details, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Simple Forecast Summary")
    lines.append("")
    lines.append(f"- Run tag: {run_tag}")
    lines.append(f"- Tickers requested: {len(tickers)}")
    lines.append(f"- Experiments: {int(summary_df.shape[0])}")
    lines.append(f"- Weekly windows: {int(windows_df.shape[0])}")
    lines.append("")

    if not summary_df.empty:
        best = summary_df.iloc[0]
        lines.append("## Best Configuration")
        lines.append(f"- Ticker: {best['ticker']}")
        lines.append(f"- Models: {best['models']}")
        lines.append(f"- Directional accuracy: {best['directional_accuracy']:.3f}")
        lines.append(f"- MAE: {best['mae']:.4f}")
        lines.append(f"- RMSE: {best['rmse']:.4f}")
        lines.append("")

    if skipped:
        lines.append("## Skipped")
        for row in skipped:
            lines.append(f"- {row['ticker']}: {row['reason']}")
        lines.append("")

    lines.append("## Output Files")
    lines.append("- forecast_summary.csv")
    lines.append("- forecast_windows.csv")
    lines.append("- run_details.json")
    lines.append("- simple_summary.md")

    report_md.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Simple forecast evaluator completed. Output dir: %s", out_dir)
    print(f"Done. Results saved in: {out_dir}")


if __name__ == "__main__":
    main()
