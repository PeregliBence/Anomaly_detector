"""Live forecasts are never rewritten; matured ones are scored; the store round-trips."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_prices, toy_panel
from src import live, store


@pytest.fixture
def live_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "ROOT", tmp_path)
    monkeypatch.setattr(live, "PRED_DIR", tmp_path / "predictions")
    monkeypatch.setattr(live, "SCORECARD_FILE", tmp_path / "scorecard.csv")
    monkeypatch.setattr(live, "UNIVERSE_FILE", tmp_path / "universe.csv")
    monkeypatch.setattr(live, "CANDIDATES_FILE", tmp_path / "candidates.csv")
    (tmp_path / "predictions").mkdir()
    return tmp_path


def test_refresh_ranks_candidates_by_dollar_volume_above_the_price_floor(live_dirs, monkeypatch):
    frames = {}
    for i, t in enumerate(["BIG", "MID", "SMALL", "PENNY"]):
        df = make_prices(n_days=120, seed=i)
        df["DollarVolume"] = [4e9, 3e9, 1e9, 9e9][i]
        df["Close"] = 1.0 if t == "PENNY" else df["Close"]
        frames[t] = df
    pd.DataFrame({"ticker": list(frames), "sharadar_ticker": list(frames), "as_of": "2020-01-01"}) \
        .to_csv(live.CANDIDATES_FILE, index=False)
    monkeypatch.setattr(live, "download_prices", lambda tickers, start: frames)

    out = live.refresh_universe(size=2)
    assert set(out["ticker"]) == {"BIG", "MID"}   # PENNY trades below $5
    assert live.load_universe() == ["BIG", "MID"]


def test_universe_goes_stale_after_a_quarter(live_dirs):
    pd.DataFrame({"ticker": ["A"], "sharadar_ticker": ["A"], "as_of": ["2026-01-01"]}) \
        .to_csv(live.UNIVERSE_FILE, index=False)
    assert not live.universe_is_stale(pd.Timestamp("2026-03-01"))
    assert live.universe_is_stale(pd.Timestamp("2026-04-15"))


def test_daily_never_rewrites_an_existing_forecast(live_dirs, monkeypatch):
    frames = {"A": make_prices(n_days=300), "B": make_prices(n_days=300, seed=1)}
    as_of = frames["A"].index[-1]
    existing = live_dirs / "predictions" / f"{as_of.date()}.csv"
    existing.write_text("ticker,score_h5\nA,0.9\n")

    monkeypatch.setattr(live, "load_universe", lambda: list(frames))
    monkeypatch.setattr(live, "download_prices", lambda tickers, start: frames)
    monkeypatch.setattr(live, "resolve", lambda frames: pd.DataFrame())

    def must_not_predict(*args, **kwargs):
        raise AssertionError("an existing forecast was about to be regenerated")

    monkeypatch.setattr(live, "predict", must_not_predict)
    summary = live.daily()

    assert summary["status"] == "already predicted"
    assert existing.read_text() == "ticker,score_h5\nA,0.9\n"


def test_resolve_scores_only_matured_forecasts(live_dirs, monkeypatch):
    monkeypatch.setattr(live.production, "HORIZONS", (5, 20))
    frames = {f"T{i}": make_prices(n_days=60, seed=i) for i in range(25)}
    made = frames["T0"].index[40]  # 19 trading days of history follow it
    closes = pd.DataFrame({t: f["Close"] for t, f in frames.items()})
    future = np.log(closes.iloc[45] / closes.iloc[40])
    pd.DataFrame({"ticker": future.index, "score_h5": future.values, "score_h20": future.values}) \
        .to_csv(live_dirs / "predictions" / f"{made.date()}.csv", index=False)

    card = live.resolve(frames)
    assert card["horizon"].tolist() == [5]           # 20 days have not passed yet
    assert card["ic"].iloc[0] == pytest.approx(1.0)  # scores equal to realised returns
    assert live.resolve(frames)["horizon"].tolist() == [5]  # not scored twice


def test_resolve_scores_absolute_direction_against_always_up(live_dirs, monkeypatch):
    monkeypatch.setattr(live.production, "HORIZONS", (5,))
    frames = {f"T{i}": make_prices(n_days=60, seed=i) for i in range(25)}
    made = frames["T0"].index[40]
    closes = pd.DataFrame({t: f["Close"] for t, f in frames.items()})
    rose = (np.log(closes.iloc[45] / closes.iloc[40]) > 0)
    # A perfect forecaster: probability 0.9 when the stock rose, 0.1 when it fell.
    pd.DataFrame({"ticker": rose.index, "score_h5": rose.values.astype(float),
                  "prob_up_h5": np.where(rose, 0.9, 0.1),
                  "prob_up_dr_h5": 0.6}) \
        .to_csv(live_dirs / "predictions" / f"{made.date()}.csv", index=False)

    row = live.resolve(frames).iloc[0]
    assert row["direction_accuracy"] == 1.0
    assert row["always_up_accuracy"] == pytest.approx(rose.mean())
    assert row["brier"] == pytest.approx(0.01)
    # The derived forecast said "up" for everyone, so it scores exactly like always-up.
    assert row["dr_direction_accuracy"] == pytest.approx(rose.mean())


def test_store_round_trips_and_replaces(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STORE_PATH", tmp_path / "test.db")
    X, y, fwd = toy_panel(n_dates=60)
    preds = pd.DataFrame({"score": X["x"], "y_true": y, "fwd_return": fwd})

    store.save_experiment("t", "m", 5, "cross_sectional", "point_in_time", {}, preds, {"ic_mean": 0.1})
    exp_id = store.save_experiment("t", "m", 5, "cross_sectional", "point_in_time", {}, preds)

    assert store.experiments()["id"].tolist() == [exp_id]
    back = store.predictions(exp_id)
    pd.testing.assert_series_equal(back["score"], preds["score"].sort_index(), check_names=False)
    assert len(store.daily_ic(exp_id)) == 60


def test_neural_models_survive_pickling_without_their_network_class():
    pytest.importorskip("torch")
    import pickle

    from src.neural import MLP

    X, y, _ = toy_panel(n_dates=80, n_names=30)
    model = MLP(horizon=5, seeds=1, epochs=2).fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X), rtol=1e-5)
