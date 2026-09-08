"""
Thesis research runner for the anomaly detector project.

Purpose:
- benchmark forecast direction accuracy across model sets
- evaluate anomaly detection quality
- estimate paper-trading outcomes through semester end

Usage example:
    python thesis_runner.py --tickers AAPL,MSFT,^GSPC,BTC-USD --start 2020-01-01 --semester-end 2026-12-31
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from app.anomaly_models import (
    detect_zscore,
    detect_iqr,
    detect_isolation_forest_ensemble,
    detect_lof,
    detect_volume_anomaly,
    detect_volatility_shift,
)
from app.backtesting import backtest_anomalies, walk_forward_forecast
from app.config import CONFIG, logger
from app.data_loader import load_data, validate_data
from app.ensemble import compute_ensemble_score
from app.features import add_all_features, prepare_ml_features
from app.forecasting import (
    ForecastResult,
    HAS_STATSMODELS,
    HAS_TF,
    HAS_XGBOOST,
    classify_direction,
    forecast_ensemble,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run thesis-grade evaluation pipeline")
    parser.add_argument("--tickers", default=None, help="Comma-separated Yahoo tickers")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD), default: today")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon in trading days")
    parser.add_argument("--semester-end", default=None, help="Semester end date (YYYY-MM-DD)")
    parser.add_argument(
        "--model-sets",
        default=None,
        help="Model sets split by ';' and model names split by ','",
    )
    parser.add_argument("--out-dir", default=None, help="Output folder")
    parser.add_argument("--profile", default="", help="Optional JSON profile file")
    parser.add_argument("--run-tag", default="", help="Optional run tag for traceability")
    parser.add_argument("--focus", default=None, help="forecast-only or full")
    parser.add_argument("--min-confidence", type=float, default=None, help="Minimum confidence to take a paper trade")
    parser.add_argument("--fee-bps", type=float, default=None, help="Per-side fee in basis points")
    parser.add_argument("--slippage-bps", type=float, default=None, help="Per-side slippage in basis points")
    parser.add_argument("--benchmark-seed", type=int, default=None, help="Random seed for benchmark strategies")
    parser.add_argument("--selection-min-trades", type=int, default=None, help="Minimum trades for GO decision")
    parser.add_argument("--selection-min-sharpe", type=float, default=None, help="Minimum Sharpe for GO decision")
    parser.add_argument("--selection-max-drawdown-pct", type=float, default=None, help="Maximum absolute drawdown percent for GO")
    parser.add_argument("--selection-min-diracc", type=float, default=None, help="Minimum directional accuracy for GO")
    return parser.parse_args()


def _parse_model_sets(text: str) -> list[list[str]]:
    model_sets = []
    for chunk in text.split(";"):
        names = [x.strip().lower() for x in chunk.split(",") if x.strip()]
        if names:
            model_sets.append(names)
    return model_sets


def _add_anomaly_columns(df: pd.DataFrame, contamination: float) -> pd.DataFrame:
    df = add_all_features(df, include_targets=True)
    X_raw, X_scaled, _, _, ml_index = prepare_ml_features(df)
    if len(ml_index) == 0:
        return pd.DataFrame()

    df = df.loc[ml_index].copy()

    df = detect_zscore(df, threshold=CONFIG.zscore_threshold, method="rolling", window=CONFIG.zscore_window)
    df = detect_iqr(df)

    if len(X_raw) >= 30:
        df = detect_isolation_forest_ensemble(
            df,
            X_scaled,
            contamination=contamination,
            n_estimators=CONFIG.if_n_estimators,
            n_runs=CONFIG.if_ensemble_runs,
        )
        df = detect_lof(df, X_scaled, contamination=contamination, n_neighbors=CONFIG.lof_n_neighbors)
    else:
        df["anomaly_iso"] = 0
        df["iso_score"] = np.nan
        df["anomaly_lof"] = 0
        df["lof_score"] = np.nan

    df = detect_volume_anomaly(df, threshold=3.0)
    df = detect_volatility_shift(df)

    df = compute_ensemble_score(
        df,
        contamination=contamination,
        vote_k=CONFIG.ensemble_vote_k,
        combine=CONFIG.ensemble_combine,
    )
    return df


def _prepare_forecast_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = add_all_features(raw_df, include_targets=True)
    _, _, _, _, ml_index = prepare_ml_features(df)
    if len(ml_index) == 0:
        return pd.DataFrame()
    return df.loc[ml_index].copy()


def _run_forecast_backtest(df: pd.DataFrame, ticker: str, models: list[str], horizon: int):
    availability = {
        "arima": HAS_STATSMODELS,
        "xgboost": HAS_XGBOOST,
        "lstm": HAS_TF,
    }
    active_models = [m for m in models if availability.get(m, False)]

    def fallback_forecast(train_df: pd.DataFrame) -> ForecastResult | None:
        if train_df.empty or "Close" not in train_df.columns:
            return None

        last_price = float(train_df["Close"].iloc[-1])
        if last_price <= 0:
            return None

        direction_info = classify_direction(train_df, horizon=horizon)
        direction = direction_info.get("direction", "sideways")

        drift = float(train_df.get("log_return", pd.Series([0.0])).tail(20).mean())
        drift = float(np.clip(drift, -0.01, 0.01))

        if direction == "up":
            step_ret = abs(drift) if abs(drift) > 1e-6 else 0.001
        elif direction == "down":
            step_ret = -abs(drift) if abs(drift) > 1e-6 else -0.001
        else:
            step_ret = 0.0

        returns = np.full(horizon, step_ret, dtype=float)
        values = last_price * np.exp(np.cumsum(returns))
        band = max(0.01, abs(step_ret) * 1.5)
        lower = values * (1.0 - band)
        upper = values * (1.0 + band)

        future_dates = pd.bdate_range(train_df.index[-1] + pd.Timedelta(days=1), periods=horizon)

        probs = direction_info.get("probabilities", {})
        confidence = float(max(probs.values())) if probs else 0.5

        return ForecastResult(
            dates=list(future_dates),
            values=values,
            lower=lower,
            upper=upper,
            direction=direction,
            confidence=round(confidence, 3),
            model_name="HeuristicFallback",
            metadata={"reason": "model_unavailable"},
        )

    def forecast_fn(train_df: pd.DataFrame):
        result = None
        if active_models:
            result, _ = forecast_ensemble(train_df, horizon=horizon, models=active_models)
        if result is None:
            return fallback_forecast(train_df)
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


def _window_returns(bt, use_net: bool = True) -> list[float]:
    if not bt.window_results:
        return []
    key = "net_signal_return_pct" if use_net else "signal_return_pct"
    return [float(w.get(key, 0.0)) / 100.0 for w in bt.window_results]


def _apply_trade_rules(
    bt,
    horizon: int,
    min_confidence: float,
    fee_bps: float,
    slippage_bps: float,
) -> dict:
    if bt.n_windows == 0:
        return {
            "trade_count": 0,
            "trade_rate": 0.0,
            "win_rate": 0.0,
            "round_trip_cost_bps": round(2.0 * (fee_bps + slippage_bps), 3),
        }

    round_trip_cost = 2.0 * (fee_bps + slippage_bps) / 10000.0
    adjusted_returns = []
    trade_count = 0
    wins = 0

    for w in bt.window_results:
        pred = str(w.get("predicted_dir", "sideways"))
        conf = float(w.get("confidence", 0.0))
        gross = float(w.get("signal_return_pct", 0.0)) / 100.0

        trade_taken = pred in ("up", "down") and conf >= min_confidence
        if trade_taken:
            trade_count += 1
            net = gross - round_trip_cost
            if net > 0:
                wins += 1
        else:
            net = 0.0

        w["trade_taken"] = int(trade_taken)
        w["net_signal_return_pct"] = round(net * 100.0, 4)
        adjusted_returns.append(net)

    equity = [1.0]
    for r in adjusted_returns:
        equity.append(equity[-1] * (1.0 + r))

    bt.equity_curve = equity

    if adjusted_returns:
        mean_ret = float(np.mean(adjusted_returns))
        std_ret = float(np.std(adjusted_returns))
        bt.mean_return_following_signal = mean_ret * 100.0
        bt.sharpe_ratio = float(mean_ret / (std_ret + 1e-9) * np.sqrt(252 / max(horizon, 1)))

        gains = sum(r for r in adjusted_returns if r > 0)
        losses = abs(sum(r for r in adjusted_returns if r < 0))
        bt.profit_factor = float(gains / (losses + 1e-9))

        eq = np.array(equity)
        peak = np.maximum.accumulate(eq)
        drawdowns = (eq - peak) / (peak + 1e-9)
        bt.max_drawdown = float(drawdowns.min())

    trade_rate = trade_count / max(bt.n_windows, 1)
    win_rate = wins / max(trade_count, 1)
    return {
        "trade_count": int(trade_count),
        "trade_rate": float(trade_rate),
        "win_rate": float(win_rate),
        "round_trip_cost_bps": round(2.0 * (fee_bps + slippage_bps), 3),
    }


def _estimate_semester_projection(bt, horizon: int, semester_end: str) -> dict:
    if bt.n_windows == 0:
        return {
            "future_windows": 0,
            "projected_return_pct": 0.0,
            "projected_equity": 1.0,
        }

    today = pd.Timestamp(date.today())
    sem_end = pd.Timestamp(semester_end)
    if sem_end <= today:
        return {
            "future_windows": 0,
            "projected_return_pct": 0.0,
            "projected_equity": 1.0,
        }

    business_days_left = len(pd.bdate_range(today + pd.Timedelta(days=1), sem_end))
    future_windows = max(0, business_days_left // max(horizon, 1))

    signal_returns = _window_returns(bt, use_net=True)
    if not signal_returns:
        return {
            "future_windows": future_windows,
            "projected_return_pct": 0.0,
            "projected_equity": 1.0,
        }

    mean_ret = float(np.mean(signal_returns))
    projected_equity = float((1.0 + mean_ret) ** future_windows)

    return {
        "future_windows": int(future_windows),
        "projected_return_pct": round((projected_equity - 1.0) * 100.0, 2),
        "projected_equity": round(projected_equity, 4),
    }


def _metrics_from_window_returns(returns: list[float], horizon: int) -> dict:
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))

    mean_ret = float(np.mean(returns)) if returns else 0.0
    std_ret = float(np.std(returns)) if returns else 0.0
    sharpe = float(mean_ret / (std_ret + 1e-9) * np.sqrt(252 / max(horizon, 1))) if returns else 0.0

    gains = sum(r for r in returns if r > 0)
    losses = abs(sum(r for r in returns if r < 0))
    profit_factor = float(gains / (losses + 1e-9)) if returns else 0.0

    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    drawdowns = (eq - peak) / (peak + 1e-9)
    max_dd = float(drawdowns.min()) if returns else 0.0

    return {
        "equity_curve": equity,
        "historical_paper_return_pct": round((equity[-1] - 1.0) * 100.0, 2),
        "mean_signal_return_pct": round(mean_ret * 100.0, 4),
        "sharpe_ratio": sharpe,
        "profit_factor": profit_factor,
        "max_drawdown_pct": round(max_dd * 100.0, 4),
    }


def _build_benchmark_rows(
    bt,
    ticker: str,
    semester_end: str,
    horizon: int,
    fee_bps: float,
    slippage_bps: float,
    seed: int,
) -> list[dict]:
    if bt.n_windows == 0:
        return []

    actual_returns = [float(w.get("actual_return_pct", 0.0)) / 100.0 for w in bt.window_results]
    actual_dirs = [str(w.get("actual_dir", "sideways")) for w in bt.window_results]
    n = len(actual_returns)
    if n == 0:
        return []

    round_trip_cost = 2.0 * (fee_bps + slippage_bps) / 10000.0
    round_trip_cost_bps = round(2.0 * (fee_bps + slippage_bps), 3)

    rng = np.random.default_rng(int(seed))
    random_pred = rng.choice(np.array(["up", "down"]), size=n, replace=True).tolist()

    strategy_defs = [
        ("benchmark_always_long", ["up"] * n),
        ("benchmark_always_short", ["down"] * n),
        ("benchmark_random_50", random_pred),
    ]

    rows = []
    for name, preds in strategy_defs:
        signal_returns = []
        correct = 0
        for pred, actual_r, actual_d in zip(preds, actual_returns, actual_dirs):
            if pred == actual_d:
                correct += 1
            gross = actual_r if pred == "up" else -actual_r
            signal_returns.append(gross - round_trip_cost)

        metrics = _metrics_from_window_returns(signal_returns, horizon=horizon)
        wins = sum(1 for r in signal_returns if r > 0)
        projection_returns = list(signal_returns)
        proj_equity = 1.0
        if projection_returns:
            mean_ret = float(np.mean(projection_returns))
            today = pd.Timestamp(date.today())
            sem_end = pd.Timestamp(semester_end)
            if sem_end > today:
                business_days_left = len(pd.bdate_range(today + pd.Timedelta(days=1), sem_end))
                future_windows = max(0, business_days_left // max(horizon, 1))
            else:
                future_windows = 0
            proj_equity = float((1.0 + mean_ret) ** future_windows)
            projected_return_pct = round((proj_equity - 1.0) * 100.0, 2)
        else:
            future_windows = 0
            projected_return_pct = 0.0

        rows.append(
            {
                "run_tag": "",
                "ticker": ticker,
                "models": name,
                "row_type": "benchmark",
                "n_windows": int(n),
                "directional_accuracy": round(correct / max(n, 1), 4),
                "mae": np.nan,
                "rmse": np.nan,
                "mean_signal_return_pct": metrics["mean_signal_return_pct"],
                "sharpe_ratio": round(metrics["sharpe_ratio"], 4),
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "profit_factor": round(metrics["profit_factor"], 4),
                "trade_count": int(n),
                "trade_rate": 1.0,
                "win_rate": round(wins / max(n, 1), 4),
                "round_trip_cost_bps": round_trip_cost_bps,
                "min_confidence": 0.0,
                "historical_paper_return_pct": metrics["historical_paper_return_pct"],
                "semester_end": semester_end,
                "projected_windows": int(future_windows),
                "projected_return_pct": projected_return_pct,
                "selection_status": "benchmark",
                "selection_reason": "baseline_strategy",
                "selection_score": np.nan,
            }
        )

    return rows


def _select_models_per_ticker(
    forecast_df: pd.DataFrame,
    min_trades: int,
    min_sharpe: float,
    max_drawdown_pct: float,
    min_diracc: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if forecast_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = forecast_df.copy()
    if "row_type" not in work.columns:
        work["row_type"] = "model"

    model_df = work[work["row_type"] == "model"].copy()
    bench_df = work[work["row_type"] == "benchmark"].copy()
    if model_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    model_df["max_drawdown_abs_pct"] = model_df["max_drawdown_pct"].abs()
    model_df["selection_score"] = (
        model_df["historical_paper_return_pct"]
        + 10.0 * model_df["directional_accuracy"]
        + 2.0 * model_df["sharpe_ratio"]
        - 0.35 * model_df["max_drawdown_abs_pct"]
    )

    bench_ref = (
        bench_df[bench_df["models"] == "benchmark_always_long"][
            ["ticker", "historical_paper_return_pct", "sharpe_ratio", "max_drawdown_pct"]
        ]
        .rename(
            columns={
                "historical_paper_return_pct": "bench_hist_return_pct",
                "sharpe_ratio": "bench_sharpe",
                "max_drawdown_pct": "bench_max_drawdown_pct",
            }
        )
        .copy()
    )
    if not bench_ref.empty:
        model_df = model_df.merge(bench_ref, on="ticker", how="left")
    else:
        model_df["bench_hist_return_pct"] = np.nan
        model_df["bench_sharpe"] = np.nan
        model_df["bench_max_drawdown_pct"] = np.nan

    per_ticker_rows = []
    selection_rows = []
    for ticker, group in model_df.groupby("ticker"):
        best = group.sort_values(
            ["selection_score", "historical_paper_return_pct", "sharpe_ratio"],
            ascending=[False, False, False],
        ).iloc[0].copy()

        reasons = []
        go = True
        if int(best.get("trade_count", 0)) < int(min_trades):
            go = False
            reasons.append(f"trade_count<{int(min_trades)}")
        if float(best.get("sharpe_ratio", 0.0)) < float(min_sharpe):
            go = False
            reasons.append(f"sharpe<{float(min_sharpe):.2f}")
        if float(abs(best.get("max_drawdown_pct", 0.0))) > float(max_drawdown_pct):
            go = False
            reasons.append(f"maxDD>{float(max_drawdown_pct):.2f}%")
        if float(best.get("directional_accuracy", 0.0)) < float(min_diracc):
            go = False
            reasons.append(f"dirAcc<{float(min_diracc):.2f}")

        bench_hist = best.get("bench_hist_return_pct", np.nan)
        if not pd.isna(bench_hist) and float(best.get("historical_paper_return_pct", -1e9)) <= float(bench_hist):
            go = False
            reasons.append("no_edge_vs_always_long")

        best["selection_status"] = "GO" if go else "NO-GO"
        best["selection_reason"] = "ok" if go else ";".join(reasons)
        per_ticker_rows.append(best.to_dict())

        selection_rows.append(
            {
                "ticker": ticker,
                "selected_models": best.get("models", ""),
                "selection_status": best["selection_status"],
                "selection_reason": best["selection_reason"],
                "selection_score": round(float(best.get("selection_score", 0.0)), 4),
                "historical_paper_return_pct": round(float(best.get("historical_paper_return_pct", 0.0)), 2),
                "benchmark_always_long_return_pct": round(float(bench_hist), 2) if not pd.isna(bench_hist) else np.nan,
                "sharpe_ratio": round(float(best.get("sharpe_ratio", 0.0)), 4),
                "directional_accuracy": round(float(best.get("directional_accuracy", 0.0)), 4),
                "max_drawdown_pct": round(float(best.get("max_drawdown_pct", 0.0)), 4),
                "trade_count": int(best.get("trade_count", 0)),
            }
        )

    selected_df = pd.DataFrame(selection_rows)
    ranked_df = pd.DataFrame(per_ticker_rows)
    return selected_df, ranked_df


def _safe(v, nd=4):
    if v is None:
        return 0.0
    try:
        return round(float(v), nd)
    except Exception:
        return 0.0


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
    if args.semester_end is None:
        args.semester_end = str(profile.get("semester_end", "2026-12-31"))
    if args.focus is None:
        args.focus = str(profile.get("focus", "forecast-only"))
    if args.min_confidence is None:
        args.min_confidence = float(profile.get("min_confidence", 0.5))
    if args.fee_bps is None:
        args.fee_bps = float(profile.get("fee_bps", 2.0))
    if args.slippage_bps is None:
        args.slippage_bps = float(profile.get("slippage_bps", 3.0))
    if args.benchmark_seed is None:
        args.benchmark_seed = int(profile.get("benchmark_seed", 42))
    if args.selection_min_trades is None:
        args.selection_min_trades = int(profile.get("selection_min_trades", 15))
    if args.selection_min_sharpe is None:
        args.selection_min_sharpe = float(profile.get("selection_min_sharpe", 0.2))
    if args.selection_max_drawdown_pct is None:
        args.selection_max_drawdown_pct = float(profile.get("selection_max_drawdown_pct", 35.0))
    if args.selection_min_diracc is None:
        args.selection_min_diracc = float(profile.get("selection_min_diracc", 0.5))

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

    if profile.get("out_dir") and args.out_dir is None:
        args.out_dir = str(profile.get("out_dir"))

    if profile.get("run_tag") and not args.run_tag:
        args.run_tag = str(profile.get("run_tag"))

    return args


def main() -> None:
    args = _parse_args()
    args = _apply_profile(args)

    if args.tickers is None:
        args.tickers = "AAPL,MSFT,NVDA,^GSPC,BTC-USD"
    if args.start is None:
        args.start = "2020-01-01"
    if args.horizon is None:
        args.horizon = 5
    if args.semester_end is None:
        args.semester_end = "2026-12-31"
    if args.model_sets is None:
        args.model_sets = "arima;xgboost;arima,xgboost;arima,xgboost,lstm"
    if args.out_dir is None:
        args.out_dir = "results/thesis"
    if args.focus is None:
        args.focus = "forecast-only"
    if args.min_confidence is None:
        args.min_confidence = 0.5
    if args.fee_bps is None:
        args.fee_bps = 2.0
    if args.slippage_bps is None:
        args.slippage_bps = 3.0
    if args.benchmark_seed is None:
        args.benchmark_seed = 42
    if args.selection_min_trades is None:
        args.selection_min_trades = 15
    if args.selection_min_sharpe is None:
        args.selection_min_sharpe = 0.2
    if args.selection_max_drawdown_pct is None:
        args.selection_max_drawdown_pct = 35.0
    if args.selection_min_diracc is None:
        args.selection_min_diracc = 0.5

    args.focus = str(args.focus).strip().lower()
    if args.focus not in ("forecast-only", "full"):
        logger.warning("Unknown focus '%s', using forecast-only", args.focus)
        args.focus = "forecast-only"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_tag = args.run_tag.strip() or pd.Timestamp.now().strftime("run-%Y%m%d-%H%M%S")

    tickers = [x.strip() for x in args.tickers.split(",") if x.strip()]
    model_sets = _parse_model_sets(args.model_sets)

    forecast_rows = []
    anomaly_rows = []
    skipped = []
    processed_tickers = 0

    logger.info("Starting thesis runner for %d tickers", len(tickers))

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

        if args.focus == "full":
            df = _add_anomaly_columns(raw, contamination=CONFIG.if_contamination)
        else:
            df = _prepare_forecast_frame(raw)

        min_rows = max(220, 200 + args.horizon * 2)
        if df.empty or len(df) < min_rows:
            skipped.append({"ticker": ticker, "reason": "insufficient_feature_rows"})
            continue

        processed_tickers += 1

        if args.focus == "full":
            anom = backtest_anomalies(df)
            anomaly_rows.append(
                {
                    "run_tag": run_tag,
                    "ticker": ticker,
                    "rows": len(df),
                    "anomaly_precision": _safe(anom.get("precision")),
                    "anomaly_recall": _safe(anom.get("recall")),
                    "anomaly_f1": _safe(anom.get("f1")),
                    "anomaly_count": int(anom.get("total_anomalies", 0)),
                    "avg_post_anomaly_return": _safe(anom.get("avg_post_anomaly_return")),
                    "significant_moves": int(anom.get("total_significant_moves", 0)),
                }
            )

        benchmark_ref_bt = None
        for idx_model, model_set in enumerate(model_sets):
            bt = _run_forecast_backtest(df, ticker=ticker, models=model_set, horizon=args.horizon)
            if idx_model == 0:
                benchmark_ref_bt = bt
            trade_stats = _apply_trade_rules(
                bt,
                horizon=args.horizon,
                min_confidence=float(args.min_confidence),
                fee_bps=float(args.fee_bps),
                slippage_bps=float(args.slippage_bps),
            )
            projection = _estimate_semester_projection(bt, horizon=args.horizon, semester_end=args.semester_end)

            hist_equity = bt.equity_curve[-1] if bt.equity_curve else 1.0
            hist_return_pct = (hist_equity - 1.0) * 100.0

            forecast_rows.append(
                {
                    "run_tag": run_tag,
                    "ticker": ticker,
                    "models": ",".join(model_set),
                    "row_type": "model",
                    "n_windows": int(bt.n_windows),
                    "directional_accuracy": _safe(bt.directional_accuracy),
                    "mae": _safe(bt.mae),
                    "rmse": _safe(bt.rmse),
                    "mean_signal_return_pct": _safe(bt.mean_return_following_signal),
                    "sharpe_ratio": _safe(bt.sharpe_ratio),
                    "max_drawdown_pct": _safe(bt.max_drawdown * 100.0),
                    "profit_factor": _safe(bt.profit_factor),
                    "trade_count": int(trade_stats["trade_count"]),
                    "trade_rate": _safe(trade_stats["trade_rate"]),
                    "win_rate": _safe(trade_stats["win_rate"]),
                    "round_trip_cost_bps": _safe(trade_stats["round_trip_cost_bps"]),
                    "min_confidence": _safe(args.min_confidence),
                    "historical_paper_return_pct": round(hist_return_pct, 2),
                    "semester_end": args.semester_end,
                    "projected_windows": projection["future_windows"],
                    "projected_return_pct": projection["projected_return_pct"],
                    "selection_status": "pending",
                    "selection_reason": "",
                    "selection_score": np.nan,
                }
            )

        # Add naive benchmark strategies on identical windows for apples-to-apples comparison.
        if benchmark_ref_bt is not None:
            bench_rows = _build_benchmark_rows(
            benchmark_ref_bt,
                ticker=ticker,
                semester_end=args.semester_end,
                horizon=args.horizon,
                fee_bps=float(args.fee_bps),
                slippage_bps=float(args.slippage_bps),
                seed=int(args.benchmark_seed) + sum(ord(ch) for ch in ticker),
            )
            for row in bench_rows:
                row["run_tag"] = run_tag
                forecast_rows.append(row)

    forecast_df = pd.DataFrame(forecast_rows)
    anomaly_df = pd.DataFrame(anomaly_rows)

    if not forecast_df.empty:
        forecast_df = forecast_df.sort_values(["ticker", "directional_accuracy", "sharpe_ratio"], ascending=[True, False, False])
    if not anomaly_df.empty:
        anomaly_df = anomaly_df.sort_values(["anomaly_f1", "anomaly_precision"], ascending=[False, False])

    forecast_csv = out_dir / "forecast_benchmark.csv"
    anomaly_csv = out_dir / "anomaly_benchmark.csv"
    summary_md = out_dir / "research_summary.md"
    details_json = out_dir / "run_details.json"
    selected_csv = out_dir / "selected_models.csv"
    decisions_md = out_dir / "deployment_decisions.md"

    selected_df, ranked_df = _select_models_per_ticker(
        forecast_df,
        min_trades=int(args.selection_min_trades),
        min_sharpe=float(args.selection_min_sharpe),
        max_drawdown_pct=float(args.selection_max_drawdown_pct),
        min_diracc=float(args.selection_min_diracc),
    )

    if not selected_df.empty and not forecast_df.empty:
        sel_map = {}
        for _, r in ranked_df.iterrows():
            key = (str(r.get("ticker", "")), str(r.get("models", "")))
            sel_map[key] = (
                str(r.get("selection_status", "pending")),
                str(r.get("selection_reason", "")),
                float(r.get("selection_score", np.nan)),
            )

        for idx, row in forecast_df.iterrows():
            if str(row.get("row_type", "model")) != "model":
                continue
            key = (str(row.get("ticker", "")), str(row.get("models", "")))
            if key in sel_map:
                status, reason, score = sel_map[key]
                forecast_df.at[idx, "selection_status"] = status
                forecast_df.at[idx, "selection_reason"] = reason
                forecast_df.at[idx, "selection_score"] = score

    forecast_df.to_csv(forecast_csv, index=False)
    anomaly_df.to_csv(anomaly_csv, index=False)
    selected_df.to_csv(selected_csv, index=False)

    details = {
        "run_tag": run_tag,
        "args": vars(args),
        "n_tickers": len(tickers),
        "n_processed_tickers": int(processed_tickers),
        "n_forecast_rows": int(forecast_df.shape[0]),
        "n_selected": int(selected_df.shape[0]),
        "focus": args.focus,
        "skipped": skipped,
        "selection_policy": {
            "min_trades": int(args.selection_min_trades),
            "min_sharpe": float(args.selection_min_sharpe),
            "max_drawdown_pct": float(args.selection_max_drawdown_pct),
            "min_diracc": float(args.selection_min_diracc),
        },
    }
    details_json.write_text(json.dumps(details, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Thesis Research Summary")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append(f"- Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Run tag: {run_tag}")
    lines.append(f"- Focus: {args.focus}")
    lines.append(f"- Min confidence for trade: {float(args.min_confidence):.2f}")
    lines.append(f"- Round-trip cost: {2.0 * (float(args.fee_bps) + float(args.slippage_bps)):.2f} bps")
    lines.append(f"- Tickers requested: {len(tickers)}")
    lines.append(f"- Tickers evaluated: {int(processed_tickers)}")
    lines.append(f"- Forecast experiments: {int(forecast_df.shape[0])}")
    lines.append("")

    if not forecast_df.empty:
        models_only = forecast_df[forecast_df["row_type"] == "model"] if "row_type" in forecast_df.columns else forecast_df
        best_global = models_only.sort_values(["directional_accuracy", "sharpe_ratio"], ascending=[False, False]).iloc[0]
        lines.append("## Best Forecast Configuration")
        lines.append(f"- Ticker: {best_global['ticker']}")
        lines.append(f"- Models: {best_global['models']}")
        lines.append(f"- Directional accuracy: {best_global['directional_accuracy']:.3f}")
        lines.append(f"- Sharpe ratio: {best_global['sharpe_ratio']:.3f}")
        lines.append(f"- Historical paper return: {best_global['historical_paper_return_pct']:.2f}%")
        lines.append(f"- Projected return to semester end: {best_global['projected_return_pct']:.2f}%")
        lines.append("")

    if not selected_df.empty:
        go_count = int((selected_df["selection_status"] == "GO").sum())
        nogo_count = int((selected_df["selection_status"] == "NO-GO").sum())
        lines.append("## Deployment Decision Snapshot")
        lines.append(f"- GO tickers: {go_count}")
        lines.append(f"- NO-GO tickers: {nogo_count}")
        lines.append(f"- Selection gates: trades>={int(args.selection_min_trades)}, sharpe>={float(args.selection_min_sharpe):.2f}, |maxDD|<={float(args.selection_max_drawdown_pct):.2f}%, dirAcc>={float(args.selection_min_diracc):.2f}")
        lines.append("")

    if not anomaly_df.empty:
        best_anom = anomaly_df.sort_values(["anomaly_f1", "anomaly_precision"], ascending=[False, False]).iloc[0]
        lines.append("## Best Anomaly Result")
        lines.append(f"- Ticker: {best_anom['ticker']}")
        lines.append(f"- F1: {best_anom['anomaly_f1']:.3f}")
        lines.append(f"- Precision: {best_anom['anomaly_precision']:.3f}")
        lines.append(f"- Recall: {best_anom['anomaly_recall']:.3f}")
        lines.append("")
    elif args.focus != "full":
        lines.append("## Anomaly Evaluation")
        lines.append("- Disabled in forecast-only mode.")
        lines.append("")

    if skipped:
        lines.append("## Skipped Tickers")
        for row in skipped:
            lines.append(f"- {row['ticker']}: {row['reason']}")
        lines.append("")

    lines.append("## Output Files")
    lines.append("- forecast_benchmark.csv")
    lines.append("- anomaly_benchmark.csv")
    lines.append("- selected_models.csv")
    lines.append("- deployment_decisions.md")
    lines.append("- run_details.json")
    lines.append("- research_summary.md")

    summary_md.write_text("\n".join(lines), encoding="utf-8")

    if selected_df.empty:
        decision_lines = [
            "# Deployment Decisions",
            "",
            "No model selection output is available (no forecast results).",
        ]
    else:
        decision_lines = [
            "# Deployment Decisions",
            "",
            "## Gate Policy",
            f"- min trades: {int(args.selection_min_trades)}",
            f"- min Sharpe: {float(args.selection_min_sharpe):.2f}",
            f"- max absolute drawdown (%): {float(args.selection_max_drawdown_pct):.2f}",
            f"- min directional accuracy: {float(args.selection_min_diracc):.2f}",
            "",
            "## Decisions",
            "",
            "| Ticker | Selected Models | Status | Reason | Hist Return% | Bench (Always Long) Return% | Sharpe | DirAcc | MaxDD% | Trades |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in selected_df.sort_values(["selection_status", "ticker"], ascending=[False, True]).iterrows():
            decision_lines.append(
                "| {ticker} | {models} | {status} | {reason} | {ret:.2f} | {bench} | {sh:.4f} | {acc:.2f}% | {dd:.2f} | {trades} |".format(
                    ticker=r["ticker"],
                    models=r["selected_models"],
                    status=r["selection_status"],
                    reason=r["selection_reason"],
                    ret=float(r["historical_paper_return_pct"]),
                    bench=(f"{float(r['benchmark_always_long_return_pct']):.2f}" if not pd.isna(r["benchmark_always_long_return_pct"]) else "n/a"),
                    sh=float(r["sharpe_ratio"]),
                    acc=float(r["directional_accuracy"]) * 100.0,
                    dd=float(r["max_drawdown_pct"]),
                    trades=int(r["trade_count"]),
                )
            )

    decisions_md.write_text("\n".join(decision_lines), encoding="utf-8")

    logger.info("Thesis runner completed. Output dir: %s", out_dir)
    print(f"Done. Results saved in: {out_dir}")


if __name__ == "__main__":
    main()
