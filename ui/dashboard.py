"""Web interface: today's forecasts, the live track record, and the research.

    streamlit run ui/dashboard.py

The live pages read the committed files under live/. The research page reads
the local research store (data/research.db) when it exists; elsewhere — on a
hosted deployment, say — it falls back to the summaries exported to
research/ by `python -m src.cli export-research`. The store itself derives
from the licensed Sharadar snapshot and never leaves the machine that holds it.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "live" / "predictions"
SCORECARD = ROOT / "live" / "scorecard.csv"
STORE = Path(os.environ.get("RESEARCH_DB", ROOT / "data" / "research.db"))
RESEARCH = ROOT / "research"
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
HORIZONS = (5, 20)

MODEL_LABELS = {
    "random": "Véletlen", "constant": "Konstans", "cs_momentum": "Momentum (12-1)",
    "cs_reversal": "Rövid távú visszafordulás", "cs_low_vol": "Alacsony volatilitás",
    "logistic": "Logisztikus regresszió", "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "mlp": "MLP", "lstm": "LSTM",
}
BASELINES = {"random", "constant", "cs_momentum", "cs_reversal", "cs_low_vol"}

st.set_page_config(page_title="Tőzsdei előrejelző", layout="wide")


def label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def prediction_files() -> list[Path]:
    return sorted(PRED_DIR.glob("*.csv")) if PRED_DIR.exists() else []


@st.cache_data(ttl=600)
def read_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(ttl=600)
def rank_history(ticker: str) -> pd.DataFrame:
    rows = []
    for path in prediction_files():
        df = read_predictions(path)
        hit = df[df["ticker"] == ticker]
        if not hit.empty:
            rows.append({"date": pd.Timestamp(path.stem), **hit.iloc[0].to_dict()})
    return pd.DataFrame(rows)


@st.cache_data(ttl=600)
def scorecard() -> pd.DataFrame:
    if not SCORECARD.exists():
        return pd.DataFrame()
    df = pd.read_csv(SCORECARD, parse_dates=["date"])
    return df.sort_values(["horizon", "date"])


@st.cache_data(ttl=3600)
def experiments() -> pd.DataFrame:
    if STORE.exists():
        with sqlite3.connect(STORE) as con:
            df = pd.read_sql("SELECT * FROM experiments ORDER BY id", con)
        metrics = pd.json_normalize(df["metrics"].map(json.loads))
        return pd.concat([df.drop(columns=["metrics"]), metrics], axis=1)
    if (RESEARCH / "experiments.csv").exists():
        return pd.read_csv(RESEARCH / "experiments.csv")
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def daily_ic(experiment_id: int) -> pd.Series:
    if STORE.exists():
        with sqlite3.connect(STORE) as con:
            df = pd.read_sql("SELECT date, ic FROM daily_ic WHERE experiment_id = ? ORDER BY date",
                             con, params=(experiment_id,))
    else:
        df = pd.read_parquet(RESEARCH / "daily_ic.parquet",
                             filters=[("experiment_id", "==", experiment_id)])
    return pd.Series(df["ic"].values, index=pd.to_datetime(df["date"])).sort_index()


def research_table(results_path: str, export_name: str) -> pd.DataFrame | None:
    """A results table from the local run if present, else from the export."""
    for path in (RESULTS / results_path, RESEARCH / export_name):
        if path.exists():
            return pd.read_csv(path)
    return None


def pending_forecasts(files: list[Path], card: pd.DataFrame) -> pd.DataFrame:
    """Forecasts not yet scored, with the date their horizon is expected to end.

    Business days approximate the exchange calendar, so a holiday can push an
    evaluation back by a day.
    """
    done = set(zip(card["date"].dt.strftime("%Y-%m-%d"), card["horizon"])) if not card.empty else set()
    rows = []
    for path in files:
        for h in HORIZONS:
            if (path.stem, h) not in done:
                due = pd.Timestamp(path.stem) + pd.offsets.BDay(h)
                rows.append({"Előrejelzés napja": path.stem, "Horizont (nap)": h,
                             "Várható kiértékelés": due.strftime("%Y-%m-%d")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_today() -> None:
    files = prediction_files()
    st.title("Mai előrejelzések")
    if not files:
        st.info("Még nincs éles előrejelzés. Az első a `python -m src.cli daily` futtatása után jelenik meg.")
        return

    latest = files[-1]
    df = read_predictions(latest)
    st.caption(
        f"Előrejelzés a {latest.stem} záróárak alapján · {len(df)} részvény. "
        "A rang azt mutatja, hogy a modell szerint a részvény a következő 5 (illetve 20) "
        "kereskedési napban hogyan teljesít a mezőny többi tagjához képest — "
        "1 = legjobb. Nem azt, hogy az ára emelkedik-e."
    )

    cols = ["ticker", "rank_h5", "rank_h20", "score_h5"]
    view = df[cols].rename(columns={"ticker": "Ticker", "rank_h5": "Rang (5 nap)",
                                    "rank_h20": "Rang (20 nap)", "score_h5": "Pontszám (5 nap)"})
    left, right = st.columns(2)
    with left:
        st.subheader("Legjobb 20 (5 nap)")
        st.dataframe(view.nsmallest(20, "Rang (5 nap)"), hide_index=True, width="stretch")
    with right:
        st.subheader("Leggyengébb 20 (5 nap)")
        st.dataframe(view.nlargest(20, "Rang (5 nap)"), hide_index=True, width="stretch")

    st.subheader("Egy részvény")
    ticker = st.selectbox("Ticker", sorted(df["ticker"]), index=None, placeholder="Válassz tickert…",
                          key="ticker_lookup")
    if ticker:
        hist = rank_history(ticker)
        row = df[df["ticker"] == ticker].iloc[0]
        a, b = st.columns(2)
        a.metric("Rang, 5 nap", f"{int(row['rank_h5'])} / {len(df)}")
        b.metric("Rang, 20 nap", f"{int(row['rank_h20'])} / {len(df)}")
        if len(hist) > 1:
            fig = px.line(hist, x="date", y=["rank_h5", "rank_h20"],
                          labels={"value": "Rang (1 = legjobb)", "date": "", "variable": ""})
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width="stretch")


def page_track_record() -> None:
    st.title("Éles eredmények")
    st.caption(
        "Minden előrejelzés a kimenetele előtt került commitolásra a repóba, így ez a "
        "teljesítmény utólag nem módosítható. Az IC a pontszám és a ténylegesen "
        "realizált hozam dátumonkénti rangkorrelációja."
    )
    sc = scorecard()
    pending = pending_forecasts(prediction_files(), sc)

    if sc.empty:
        st.info("Még nincs lejárt előrejelzés. Egy előrejelzés akkor értékelődik ki, amikor "
                "letelik a horizontja (5 vagy 20 kereskedési nap); a kiértékelést az aznap esti "
                "automatikus futás végzi (22:30 UTC).")

    for h, group in (sc.groupby("horizon") if not sc.empty else []):
        st.subheader(f"{h} napos horizont")
        a, b, c, d = st.columns(4)
        a.metric("Kiértékelt napok", len(group))
        b.metric("Átlagos IC", f"{group['ic'].mean():.3f}")
        c.metric("Pozitív IC aránya", f"{(group['ic'] > 0).mean():.0%}")
        d.metric("Átlagos long-short hozam", f"{group['spread'].mean():.2%}")

        fig = go.Figure()
        fig.add_bar(x=group["date"], y=group["ic"], name="IC", marker_color="#8fb3c9")
        fig.add_scatter(x=group["date"], y=group["ic"].rolling(20, min_periods=5).mean(),
                        name="20 napos átlag", line=dict(color="#0e4f52", width=2))
        fig.add_hline(y=0, line_color="#888", line_width=1)
        fig.update_layout(height=320, margin=dict(t=10, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

    if not pending.empty:
        st.subheader("Kiértékelésre váró előrejelzések")
        st.dataframe(pending.sort_values(["Várható kiértékelés", "Horizont (nap)"]),
                     hide_index=True, width="stretch")
        st.caption("A várható dátum munkanapokkal számolt becslés; tőzsdei ünnepnap egy nappal "
                   "eltolhatja.")


def page_research() -> None:
    st.title("Kutatási eredmények")
    ex = experiments()
    if ex.empty:
        st.info("Nincs elérhető kutatási eredmény. Helyben a data/research.db, máshol a "
                "`python -m src.cli export-research` által írt research/ mappa szükséges.")
        return

    st.caption("Kiértékelés: 2008–2026, túlélési torzítástól mentes univerzum (a legforgalmasabb "
               "1000 részvény), purged walk-forward. A t-érték Newey-West korrekcióval, foldonként.")

    # The final evaluation supersedes the earlier exploratory runs where present.
    final = ex[ex["name"] == "final"]
    arch = final[final["horizon"].isin([5, 20])] if not final.empty else ex[ex["name"] == "architectures"]
    if not arch.empty:
        st.subheader("Modell-architektúrák")
        view = arch.assign(Modell=arch["model"].map(label),
                           Típus=np.where(arch["model"].isin(BASELINES), "Baseline", "Tanuló modell"))
        fig = px.bar(view, x="Modell", y="ic_mean", color="Típus", facet_col="horizon",
                     labels={"ic_mean": "Átlagos IC", "horizon": "Horizont (nap)"},
                     color_discrete_map={"Baseline": "#b9c4bf", "Tanuló modell": "#0e4f52"})
        fig.update_layout(height=380, margin=dict(t=30, b=10))
        st.plotly_chart(fig, width="stretch")

    portfolio = research_table("report/models.csv", "portfolio.csv")
    if portfolio is not None:
        st.subheader("Long-short portfólió")
        st.caption("A felső 10% vétele, az alsó 10% shortolása, egyenlő súllyal, a horizonttal "
                   "megegyező, nem átfedő tartással; tranzakciós költség nélkül. A Sharpe min–max "
                   "azt mutatja, mennyit számít, melyik napon kezdjük az újrasúlyozást.")
        view = portfolio[portfolio["variant"] == "final"].assign(Modell=lambda d: d["model"].map(label))
        view = view[["horizon", "Modell", "ic_mean", "ls_ann_return", "ls_sharpe",
                     "ls_sharpe_min", "ls_sharpe_max", "ls_max_drawdown"]].rename(columns={
            "horizon": "Horizont", "ic_mean": "IC", "ls_ann_return": "Éves hozam",
            "ls_sharpe": "Sharpe", "ls_sharpe_min": "Sharpe min", "ls_sharpe_max": "Sharpe max",
            "ls_max_drawdown": "Max. visszaesés"})
        st.dataframe(view.round(3), hide_index=True, width="stretch")

    alt = ex[ex["name"] == "final_alt"]
    if not final.empty and not alt.empty:
        st.subheader("Hozzáad-e az alternatív adat?")
        st.caption("Fundamentumok, gyorsjelentés-meglepetés és -reakció, kedvezőtlen 8-K események, "
                   "bennfentes kereskedés — ugyanaz a modell velük és nélkülük.")
        both = pd.concat([final.assign(Változat="Ár- és piaci jellemzők"),
                          alt.assign(Változat="+ alternatív adat")])
        both = both[both["model"].isin(alt["model"])]
        fig = px.bar(both.assign(Modell=both["model"].map(label)), x="Modell", y="ic_mean",
                     color="Változat", barmode="group", facet_col="horizon",
                     labels={"ic_mean": "Átlagos IC", "horizon": "Horizont (nap)"},
                     color_discrete_sequence=["#0e4f52", "#c8934f"])
        fig.update_layout(height=380, margin=dict(t=30, b=10))
        st.plotly_chart(fig, width="stretch")

    sweep = final if not final.empty else ex[ex["name"] == "horizon_sweep"]
    sweep = sweep[sweep["model"].isin(sweep.groupby("model")["horizon"].nunique()
                                      .loc[lambda s: s >= 3].index)]
    if not sweep.empty:
        st.subheader("Horizontok")
        fig = px.line(sweep.assign(Modell=sweep["model"].map(label)).sort_values("horizon"),
                      x="horizon", y="ic_mean", color="Modell", markers=True,
                      labels={"horizon": "Horizont (kereskedési nap)", "ic_mean": "Átlagos IC"})
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Egy kísérlet időben")
    options = {f"{r['name']} · {label(r['model'])} · {r['horizon']} nap": int(r["id"])
               for _, r in ex.iterrows()}
    choice = st.selectbox("Kísérlet", list(options), key="experiment_pick")
    ic = daily_ic(options[choice])
    if not ic.empty:
        yearly = ic.groupby(ic.index.year).mean()
        a, b = st.columns(2)
        with a:
            fig = px.bar(x=yearly.index, y=yearly.values, labels={"x": "", "y": "Átlagos IC"})
            fig.update_traces(marker_color=np.where(yearly.values >= 0, "#0e4f52", "#c8934f"))
            fig.update_layout(height=300, margin=dict(t=10, b=10), title="Évenként")
            st.plotly_chart(fig, width="stretch")
        with b:
            fig = px.line(x=ic.index, y=ic.cumsum().values, labels={"x": "", "y": "Kumulált IC"})
            fig.update_layout(height=300, margin=dict(t=10, b=10), title="Kumulált")
            st.plotly_chart(fig, width="stretch")

    df = research_table("survivorship/cross_sectional_by_universe.csv", "survivorship.csv")
    if df is not None:
        st.subheader("Túlélési torzítás")
        st.caption("Ugyanaz a módszer három univerzumon. A torzított univerzumokból hiányoznak "
                   "a később kivezetett cégek.")
        df["Modell"] = df["model"].map(label)
        pivot = df.pivot_table(index=["horizon", "Modell"], columns="universe", values="ic_mean")
        pivot = pivot.rename(columns={"point_in_time": "Helyes (pontbeli)",
                                      "survivors": "Csak túlélők", "todays_list": "Mai lista"})
        st.dataframe(pivot.round(4), width="stretch")

    df = research_table("survivorship/absolute_direction.csv", "absolute_direction.csv")
    if df is not None:
        st.subheader("Abszolút irány")
        st.caption("Emelkedik-e az ár? A „Konstans” modell mindig emelkedést jósol.")
        df["Modell"] = df["model"].map(label)
        view = df[["horizon", "Modell", "accuracy", "majority_baseline", "auc"]].rename(columns={
            "horizon": "Horizont", "accuracy": "Találati arány",
            "majority_baseline": "„Mindig fel” találati aránya", "auc": "AUC"})
        st.dataframe(view.round(4), hide_index=True, width="stretch")


PAGES = {
    "Mai előrejelzések": page_today,
    "Éles eredmények": page_track_record,
    "Kutatási eredmények": page_research,
}

with st.sidebar:
    st.header("Tőzsdei előrejelző")
    page = st.radio("Oldal", list(PAGES), label_visibility="collapsed", key="page")
    st.caption("5 és 20 napos keresztmetszeti előrejelzés, a legforgalmasabb 1000 amerikai részvényre.")

PAGES[page]()
