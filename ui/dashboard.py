"""Web interface: an overview, today's forecasts, the live track record and the research.

    streamlit run ui/dashboard.py

The live pages read the committed files under live/. The research pages read
the local research store (data/research.db) when it exists; elsewhere — on a
hosted deployment, say — they fall back to the summaries exported to
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
import plotly.graph_objects as go
import plotly.io as pio
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
    "mlp": "MLP", "lstm": "LSTM", "ensemble": "Ensemble",
}
BASELINES = {"random", "constant", "cs_momentum", "cs_reversal", "cs_low_vol"}
UNIVERSES = {"point_in_time": "Pontbeli (helyes)", "survivors": "Csak túlélők",
             "todays_list": "Mai lista"}
APPROACHES = {"absolute_ensemble": "Közvetlen modell", "drift_plus_rank": "Alapráta + relatív rang",
              "always_up": "„Mindig fel”"}

# One colour per role, used the same way on every page: the models being
# studied, the baselines they are measured against, and a compared variant.
MODEL, COMPARE, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
BASELINE = "#b5b3aa"
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BAND = "rgba(42,120,214,0.10)"
MARKET_TAKEAWAY = (
    "A 24 piaci oszlop (amelyek egy napon belül minden részvénynél azonosak, így együtt azonosítják "
    "a napot) elhagyásával a tanító veszteség javulása nagyjából a felére csökken, a validációs "
    "legjobb érték viszont gyakorlatilag ugyanaz. A túltanulás kb. fele tehát az egyes napok "
    "bemagolásából jön, és a piaci jellemzők hasznát a validációs veszteség nem mutatja.")
TABLE_15 = 16 * 35 + 3  # header plus 15 rows, no inner scrolling

pio.templates["thesis"] = go.layout.Template(layout=dict(
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=13, color=INK_2),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[MODEL, COMPARE, THIRD, "#eda100", "#e87ba4"],
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, ticks="", showline=True),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, ticks="", showline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
    hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK)),
    margin=dict(l=10, r=10, t=40, b=10),
    separators=", ",
    bargap=0.35,
))
pio.templates.default = "thesis"

st.set_page_config(page_title="Tőzsdei előrejelző", page_icon=":material/insights:", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1280px;}
  h1 {font-weight: 650; letter-spacing: -0.01em;}
  h3 {font-weight: 600; margin-top: 0.6rem;}
  [data-testid="stMetricValue"] {font-weight: 600;}
  .lead {color: #52514e; font-size: 1.05rem; margin: -0.4rem 0 1.4rem 0; max-width: 60rem;}
  .takeaway {border-left: 3px solid #2a78d6; padding: 0.15rem 0 0.15rem 0.9rem;
             color: #0b0b0b; margin: 0.2rem 0 1rem 0;}
  .finding h4 {margin: 0 0 0.3rem 0; font-size: 1rem; font-weight: 600;}
  .finding {min-height: 6.6rem;}
  .finding p {color: #52514e; margin: 0; font-size: 0.93rem;}
</style>
""", unsafe_allow_html=True)


def label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def hu(x: float, fmt: str = ".3f") -> str:
    """A number with a decimal comma, the Hungarian convention."""
    return format(x, fmt).replace(".", ",")


def pct(x: float, digits: int = 1) -> str:
    return hu(100 * x, f".{digits}f") + "%"


def lead(text: str) -> None:
    st.markdown(f'<p class="lead">{text}</p>', unsafe_allow_html=True)


def takeaway(text: str) -> None:
    st.markdown(f'<div class="takeaway">{text}</div>', unsafe_allow_html=True)


def chart(fig: go.Figure, height: int = 360) -> None:
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def horizon_picker(key: str, options=HORIZONS) -> int:
    return st.segmented_control("Horizont", list(options), default=options[0], key=key,
                                format_func=lambda h: f"{h} nap", label_visibility="collapsed") \
        or options[0]


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
    return pd.read_csv(SCORECARD, parse_dates=["date"]).sort_values(["horizon", "date"])


@st.cache_data(ttl=3600)
def experiments() -> pd.DataFrame:
    if STORE.exists():
        with sqlite3.connect(STORE) as con:
            df = pd.read_sql("SELECT * FROM experiments ORDER BY id", con)
        metrics = pd.json_normalize(df["metrics"].map(json.loads))
        df = pd.concat([df.drop(columns=["metrics"]), metrics], axis=1)
    elif (RESEARCH / "experiments.csv").exists():
        df = pd.read_csv(RESEARCH / "experiments.csv")
    else:
        return pd.DataFrame()
    # The metrics repeat a few experiment fields (the horizon, for one).
    df = df.loc[:, ~df.columns.duplicated()]
    return df.drop(columns=[c for c in df.columns if c.endswith(".1")])


@st.cache_data(ttl=3600)
def daily_ic(experiment_id: int) -> pd.Series:
    if STORE.exists():
        with sqlite3.connect(STORE) as con:
            df = pd.read_sql("SELECT date, ic FROM daily_ic WHERE experiment_id = ? ORDER BY date",
                             con, params=(experiment_id,))
    elif (RESEARCH / "daily_ic.parquet").exists():
        df = pd.read_parquet(RESEARCH / "daily_ic.parquet",
                             filters=[("experiment_id", "==", experiment_id)])
    else:
        return pd.Series(dtype=float)
    return pd.Series(df["ic"].values, index=pd.to_datetime(df["date"])).sort_index()


@st.cache_data(ttl=3600)
def research_table(results_path: str, export_name: str) -> pd.DataFrame | None:
    """A results table from the local run if present, else from the export."""
    for path in (RESULTS / results_path, RESEARCH / export_name):
        if path.exists():
            return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return None


def final_runs(name: str = "final") -> pd.DataFrame:
    ex = experiments()
    return ex[ex["name"] == name] if not ex.empty else ex


def experiment_id(model: str, horizon: int, name: str = "final") -> int | None:
    runs = final_runs(name)
    hit = runs[(runs["model"] == model) & (runs["horizon"] == horizon)]
    return int(hit["id"].iloc[-1]) if not hit.empty else None


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
                rows.append({"Előrejelzés napja": path.stem, "Horizont": f"{h} nap",
                             "Várható kiértékelés": due.strftime("%Y-%m-%d")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Charts shared by several pages
# ---------------------------------------------------------------------------

def model_bars(runs: pd.DataFrame, value: str = "ic_mean", title: str = "Átlagos IC",
               fmt: str = ".3f") -> go.Figure:
    """Horizontal bars, learned models in colour and baselines in grey."""
    d = runs.sort_values(value)
    colors = [BASELINE if m in BASELINES else MODEL for m in d["model"]]
    fig = go.Figure(go.Bar(
        x=d[value], y=d["model"].map(label), orientation="h", marker_color=colors,
        marker_cornerradius=4, text=d[value].map(lambda v: hu(v, fmt)), textposition="outside",
        textfont=dict(color=INK_2, size=12), cliponaxis=False,
        hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
    fig.add_vline(x=0, line_color=AXIS, line_width=1)
    fig.update_layout(xaxis_title=title, yaxis_title=None, showlegend=False,
                      xaxis=dict(showline=False), yaxis=dict(showgrid=False))
    return fig


def grouped_bars(d: pd.DataFrame, category: str, group: str, value: str, colors: dict,
                 fmt: str = ".3f", title: str = "Átlagos IC") -> go.Figure:
    fig = go.Figure()
    for name, color in colors.items():
        g = d[d[group] == name]
        fig.add_bar(x=g[category], y=g[value], name=name, marker_color=color, marker_cornerradius=4,
                    hovertemplate="%{x}<br>" + name + ": %{y:" + fmt + "}<extra></extra>")
    fig.add_hline(y=0, line_color=AXIS, line_width=1)
    fig.update_layout(barmode="group", bargroupgap=0.08, yaxis_title=title, xaxis_title=None,
                      xaxis=dict(showgrid=False))
    return fig


def legend_note() -> None:
    st.caption(f"<span style='color:{MODEL}'>■</span> tanuló modell  "
               f"<span style='color:{BASELINE}'>■</span> egyszerű szabály (baseline)",
               unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview() -> None:
    st.title("Részvényárfolyam-előrejelzés gépi tanulással")
    lead("Melyik részvény teljesít jobban a következő 5 vagy 20 kereskedési napban, és merre "
         "mozdul az ára? A modelleket 2008–2026 között, túlélési torzítástól mentes adaton és "
         "szivárgásmentes walk-forward kiértékeléssel mértük, 2026 októbere óta pedig minden nap "
         "élesben, előre rögzített előrejelzésekkel futnak.")

    portfolio = research_table("report/models.csv", "portfolio.csv")
    runs = final_runs()
    three = research_table("final/absolute_three_way.csv", "absolute_three_way.csv")
    card = scorecard()

    cols = st.columns(4)
    if portfolio is not None:
        p = portfolio[(portfolio["variant"] == "final") & (portfolio["horizon"] == 5)].set_index("model")
        if "ensemble" in p.index:
            best_base = p.loc[p.index.isin(BASELINES), "ic_mean"].max()
            cols[0].metric("Rangsorolási pontosság (IC), 5 nap", hu(p.loc["ensemble", "ic_mean"]),
                           f"+{hu(p.loc['ensemble', 'ic_mean'] - best_base)} a legjobb szabályhoz",
                           border=True, help="A modell pontszáma és a ténylegesen elért hozam napi "
                           "rangkorrelációja, 2008–2026 átlaga. 0 = semmi, 1 = tökéletes sorrend.")
            cols[1].metric("Long-short Sharpe-ráta, 5 nap", hu(p.loc["ensemble", "ls_sharpe"], ".2f"),
                           "költségek nélkül", delta_color="off", border=True,
                           help="A felső tized vétele, az alsó tized shortolása, évesítve.")
    if not card.empty:
        live = card[card["horizon"] == 5]
        cols[2].metric("Éles IC, 5 nap", hu(live["ic"].mean()) if len(live) else "–",
                       f"{len(live)} kiértékelt nap", delta_color="off", border=True,
                       help="Az élesben, előre rögzített előrejelzések átlagos IC-je.")
    if three is not None:
        t = three[three["horizon"] == 5].set_index("approach")
        cols[3].metric("Irány-találat, 5 nap", pct(t.loc["absolute_ensemble", "accuracy"]),
                       f"{hu(100 * t.loc['absolute_ensemble', 'edge_vs_always_up'], '+.1f')} pp a „mindig fel”-hez",
                       border=True, help="Emelkedik-e a részvény ára: a közvetlen modell találati "
                       "aránya a mindig emelkedést mondó szabályhoz képest (2009–2026).")

    st.subheader("Fő megállapítások")
    findings = [
        ("Gyenge, de valódi relatív jel",
         "5 és 20 napon a legjobb tanuló modellek megverik az ismert egyszerű szabályokat. Az "
         "előny kicsi, de 5 napon 2009 kivételével minden évben megjelenik, és szignifikáns."),
        ("A rövid horizont a megbízható",
         "5 napon a legerősebb és legstabilabb a jel, itt a legjobb a long-short portfólió. "
         "Hosszabb horizonton a modellek már nem verik az alacsony volatilitás szabályt."),
        ("Az árfolyam iránya nem jósolható",
         "A közvetlen irány-modell rosszabb a „mindig fel” szabálynál: megpróbálja időzíteni a "
         "piacot, és ebben vak. A relatív rangsor ennél sokkal jobban működik."),
        ("Több adat és hangolás nem segít",
         "Fundamentumok, gyorsjelentések, 8-K események és bennfentes kereskedés hozzáadása, "
         "illetve a hiperparaméter-hangolás sem javított mérhetően."),
    ]
    for row in (findings[:2], findings[2:]):
        for col, (title, text) in zip(st.columns(2), row):
            with col.container(border=True):
                st.markdown(f'<div class="finding"><h4>{title}</h4><p>{text}</p></div>',
                            unsafe_allow_html=True)

    if not runs.empty:
        st.subheader("Modellek és egyszerű szabályok")
        h = horizon_picker("overview_h")
        chart(model_bars(runs[runs["horizon"] == h]), 380)
        legend_note()


def page_today() -> None:
    files = prediction_files()
    st.title("Mai előrejelzések")
    if not files:
        st.info("Még nincs éles előrejelzés. Az első a `python -m src.cli daily` futtatása után jelenik meg.")
        return

    latest = files[-1]
    df = read_predictions(latest)
    lead(f"A {latest.stem}-i záróárak alapján, {len(df)} részvényre. A rang a mezőny többi "
         "tagjához viszonyított várható teljesítmény (1 = legjobb), a valószínűség pedig az, "
         "hogy az ár a horizont végére emelkedik.")

    h = horizon_picker("today_h")
    has_direction = f"prob_up_h{h}" in df
    if has_direction:
        a, b, c = st.columns(3)
        a.metric("Részvények száma", len(df), border=True)
        b.metric("Emelkedést vár: közvetlen modell", pct((df[f"prob_up_h{h}"] > 0.5).mean(), 0),
                 border=True)
        if f"prob_up_dr_h{h}" in df:
            c.metric("Emelkedést vár: alapráta + rang", pct((df[f"prob_up_dr_h{h}"] > 0.5).mean(), 0),
                     border=True)
        st.caption("A backtestben egyik irány-előrejelzés sem volt jobb a „mindig fel” szabálynál; "
                   "a közvetlen modell a piac egészének irányát is megpróbálja eltalálni.")

    view = pd.DataFrame({"Ticker": df["ticker"], "Rang": df[f"rank_h{h}"].astype(int),
                         "Pontszám": df[f"score_h{h}"]})
    columns = {
        "Rang": st.column_config.NumberColumn(format="%d", width="small"),
        "Pontszám": st.column_config.ProgressColumn(
            help="A modellek átlagos rangpontszáma (0–1).", min_value=0.0, max_value=1.0, format="%.2f"),
    }
    if has_direction:
        view["P(emelkedik)"] = df[f"prob_up_h{h}"]
        columns["P(emelkedik)"] = st.column_config.ProgressColumn(
            min_value=0.0, max_value=1.0, format="percent")

    left, right = st.columns(2)
    with left:
        st.subheader("A legerősebb 15")
        st.dataframe(view.nsmallest(15, "Rang"), hide_index=True, width="stretch", height=TABLE_15,
                     column_config=columns)
    with right:
        st.subheader("A leggyengébb 15")
        st.dataframe(view.nlargest(15, "Rang").sort_values("Rang"), hide_index=True, width="stretch",
                     height=TABLE_15, column_config=columns)

    st.subheader("Egy részvény")
    ticker = st.selectbox("Ticker", sorted(df["ticker"]), index=None, placeholder="Válassz tickert…",
                          key="ticker_lookup", label_visibility="collapsed")
    if ticker:
        row = df[df["ticker"] == ticker].iloc[0]
        cols = st.columns(4)
        for i, hh in enumerate(HORIZONS):
            cols[i].metric(f"Rang, {hh} nap", f"{int(row[f'rank_h{hh}'])} / {len(df)}", border=True)
            if f"prob_up_h{hh}" in row and pd.notna(row[f"prob_up_h{hh}"]):
                cols[i + 2].metric(f"P(emelkedik), {hh} nap", pct(row[f"prob_up_h{hh}"], 0), border=True)
        hist = rank_history(ticker)
        if len(hist) > 1:
            fig = go.Figure()
            for hh, color in zip(HORIZONS, (MODEL, COMPARE)):
                fig.add_scatter(x=hist["date"], y=hist[f"rank_h{hh}"], name=f"{hh} nap", mode="lines+markers",
                                line=dict(color=color, width=2), marker=dict(size=8))
            fig.update_yaxes(autorange="reversed", title="Rang (1 = legjobb)")
            chart(fig, 300)


def page_track_record() -> None:
    st.title("Éles eredmények")
    lead("Minden előrejelzés a kimenetele előtt került a repóba, így ez a teljesítmény utólag nem "
         "módosítható. Egy előrejelzés akkor értékelődik ki, amikor letelik a horizontja; ezt az "
         "aznap esti automatikus futás végzi.")
    sc = scorecard()
    pending = pending_forecasts(prediction_files(), sc)
    h = horizon_picker("live_h")
    group = sc[sc["horizon"] == h] if not sc.empty else sc

    if group.empty:
        st.info(f"Még nincs lejárt {h} napos előrejelzés.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Kiértékelt napok", len(group), border=True)
        b.metric("Átlagos IC", hu(group["ic"].mean()), border=True)
        c.metric("Pozitív IC-jű napok", pct((group["ic"] > 0).mean(), 0), border=True)
        d.metric("Átlagos long-short hozam", pct(group["spread"].mean(), 2), border=True,
                 help=f"A felső és az alsó tized {h} napos hozamának különbsége.")

        st.subheader("Napi IC a backtesthez viszonyítva")
        fig = go.Figure()
        exp = experiment_id("lightgbm", h)
        backtest = daily_ic(exp) if exp is not None else pd.Series(dtype=float)
        if not backtest.empty:
            lo, hi = backtest.quantile([0.1, 0.9])
            fig.add_hrect(y0=lo, y1=hi, fillcolor=BAND, line_width=0)
            fig.add_hline(y=backtest.mean(), line_color=MODEL, line_width=1, opacity=0.6,
                          annotation_text="backtest átlag", annotation_position="bottom right",
                          annotation_font=dict(color=INK_2, size=11))
        fig.add_bar(x=group["date"].dt.strftime("%m.%d."), y=group["ic"], name="Éles IC", marker_color=MODEL,
                    marker_cornerradius=4, hovertemplate="%{x}: %{y:.3f}<extra></extra>")
        fig.add_hline(y=0, line_color=AXIS, line_width=1)
        fig.update_layout(showlegend=False, yaxis_title="IC", bargap=0.5,
                          xaxis=dict(showgrid=False, type="category", title="Előrejelzés napja"))
        chart(fig, 340)
        if not backtest.empty:
            st.caption("A kék sáv a 2008–2026-os backtest napjainak középső 80%-a (LightGBM, azonos "
                       "jellemzők). Egy-egy hét IC-je ennél jóval szélesebb tartományban mozog, mint az "
                       "átlag; az éles eredmény csak sok hét után értelmezhető. Az átfedő horizontok "
                       "miatt az egymást követő napok nem függetlenek.")

        if "direction_accuracy" in group and group["direction_accuracy"].notna().any():
            g = group.dropna(subset=["direction_accuracy"])
            st.subheader("Abszolút irány: emelkedik-e az ár?")
            a, b, c = st.columns(3)
            a.metric("Közvetlen modell", pct(g["direction_accuracy"].mean()),
                     f"{hu(100 * (g['direction_accuracy'] - g['always_up_accuracy']).mean(), '+.1f')} pp",
                     border=True)
            if "dr_direction_accuracy" in g and g["dr_direction_accuracy"].notna().any():
                b.metric("Alapráta + relatív rang", pct(g["dr_direction_accuracy"].mean()),
                         f"{hu(100 * (g['dr_direction_accuracy'] - g['always_up_accuracy']).mean(), '+.1f')} pp",
                         border=True)
            c.metric("„Mindig fel”", pct(g["always_up_accuracy"].mean()), "viszonyítási alap",
                     delta_color="off", border=True)
            fig = go.Figure()
            # The benchmark is drawn first and widest: the derived forecast
            # often says "up" for everyone and then lies exactly on it.
            series = [("always_up_accuracy", "„Mindig fel”", BASELINE, 6, "solid"),
                      ("dr_direction_accuracy", "Alapráta + rang", COMPARE, 2, "dot"),
                      ("direction_accuracy", "Közvetlen modell", MODEL, 2, "solid")]
            for col, name, color, width, dash in series:
                if col in g:
                    fig.add_scatter(x=g["date"].dt.strftime("%m.%d."), y=g[col], name=name, mode="lines+markers",
                                    line=dict(color=color, width=width, dash=dash), marker=dict(size=8),
                                    hovertemplate="%{x}: %{y:.1%}<extra>" + name + "</extra>")
            fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Találati arány",
                              xaxis=dict(type="category", showgrid=False))
            chart(fig, 300)

    if not pending.empty:
        with st.expander(f"Kiértékelésre váró előrejelzések ({len(pending)})"):
            st.dataframe(pending.sort_values(["Várható kiértékelés", "Horizont"]),
                         hide_index=True, width="stretch")
            st.caption("A várható dátum munkanapokkal számolt becslés; tőzsdei ünnepnap eltolhatja.")


def page_research() -> None:
    st.title("Kutatási eredmények")
    if experiments().empty:
        st.info("Nincs elérhető kutatási eredmény. Helyben a data/research.db, máshol a "
                "`python -m src.cli export-research` által írt research/ mappa szükséges.")
        return
    lead("Kiértékelés 2008–2026 között, a mindenkori legforgalmasabb 1000 amerikai részvényen "
         "(a később kivezetett cégekkel együtt), évente újratanított modellekkel és purged "
         "walk-forward módszerrel: a modell sosem látja azt az időszakot, amelyen mérjük.")

    tabs = st.tabs(["Modellek", "Horizontok", "Időbeli stabilitás", "Portfólió", "Abszolút irány",
                    "Robusztusság", "Tanulási görbék"])
    with tabs[0]:
        tab_models()
    with tabs[1]:
        tab_horizons()
    with tabs[2]:
        tab_stability()
    with tabs[3]:
        tab_portfolio()
    with tabs[4]:
        tab_absolute()
    with tabs[5]:
        tab_robustness()
    with tabs[6]:
        tab_learning_curves()


def tab_models() -> None:
    runs = final_runs()
    takeaway("5 napon minden tanuló modell megveri az egyszerű szabályokat, a legerősebbek a "
             "fa-alapú modellek (LightGBM, XGBoost); a neurális hálók nem adnak többet. 20 napon "
             "kisebb a különbség.")
    h = horizon_picker("models_h")
    d = runs[runs["horizon"] == h]
    chart(model_bars(d), 400)
    legend_note()
    with st.expander("Táblázat"):
        table = d.assign(Modell=d["model"].map(label))[
            ["Modell", "ic_mean", "ic_tstat_folds", "ic_positive_rate", "auc"]]
        st.dataframe(table.sort_values("ic_mean", ascending=False), hide_index=True, width="stretch",
                     column_config={
                         "ic_mean": st.column_config.NumberColumn("Átlagos IC", format="%.4f"),
                         "ic_tstat_folds": st.column_config.NumberColumn(
                             "t-érték", format="%.2f", help="Az évenkénti foldok IC-jéből számolva."),
                         "ic_positive_rate": st.column_config.NumberColumn("Pozitív IC-jű napok",
                                                                           format="percent"),
                         "auc": st.column_config.NumberColumn("AUC", format="%.4f")})


def tab_horizons() -> None:
    runs = final_runs()
    counts = runs.groupby("model")["horizon"].nunique()
    runs = runs[runs["model"].isin(counts[counts >= 3].index)]
    if runs.empty:
        st.info("Nincs több horizontra futtatott kísérlet.")
        return
    takeaway("5 napon a LightGBM a legjobb, és itt a legmegbízhatóbb a becslés (a legmagasabb "
             "t-érték). Hosszabb horizonton kevés a független időszak, az eredmény zajos, és az "
             "alacsony volatilitás szabály utoléri, majd megelőzi a modellt.")
    metric = st.segmented_control("Mérőszám", ["ic_mean", "ic_tstat_folds"], default="ic_mean",
                                  key="hz_metric", label_visibility="collapsed",
                                  format_func={"ic_mean": "Átlagos IC", "ic_tstat_folds": "t-érték"}.get) \
        or "ic_mean"
    fig = go.Figure()
    emphasis = {"lightgbm": MODEL, "cs_low_vol": COMPARE}
    for model, g in runs.groupby("model"):
        g = g.sort_values("horizon")
        color = emphasis.get(model, BASELINE)
        fig.add_scatter(x=g["horizon"], y=g[metric], name=label(model), mode="lines+markers",
                        line=dict(color=color, width=2.5 if model in emphasis else 1.5),
                        marker=dict(size=8), opacity=1 if model in emphasis else 0.8,
                        hovertemplate="%{x} nap: %{y:.3f}<extra>" + label(model) + "</extra>")
    fig.add_hline(y=0, line_color=AXIS, line_width=1)
    fig.update_layout(xaxis=dict(type="log", tickvals=sorted(runs["horizon"].unique()), showgrid=False,
                                 title="Horizont (kereskedési nap)"),
                      yaxis_title="Átlagos IC" if metric == "ic_mean" else "t-érték (foldonként)")
    chart(fig, 380)


def tab_stability() -> None:
    yearly = research_table("report/ensemble_ic_by_year.csv", "ensemble_ic_by_year.csv")
    if yearly is not None:
        takeaway("5 napon az ensemble IC-je 2009 kivételével minden évben pozitív; 2009-ben a válság "
                 "utáni hirtelen fordulat a korábban gyenge részvényeket emelte. 20 napon ingadozóbb.")
        h = horizon_picker("stab_h")
        col = next((c for c in yearly.columns if c.startswith("Ár") and c.endswith(f"{h} nap")), None)
        if col:
            fig = go.Figure(go.Bar(x=yearly["date"], y=yearly[col], marker_color=MODEL, marker_cornerradius=4,
                                   hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
            fig.add_hline(y=0, line_color=AXIS, line_width=1)
            fig.update_layout(yaxis_title="Átlagos IC", xaxis=dict(showgrid=False, dtick=2))
            chart(fig, 340)

    st.subheader("Kumulált IC")
    st.caption("Egyenletesen emelkedő görbe: a jel időben stabil. Lapos szakasz: az adott időszakban "
               "a modell nem tudott rangsorolni.")
    h = horizon_picker("cum_h")
    options = [m for m in ("lightgbm", "xgboost", "mlp", "lstm", "logistic", "cs_reversal",
                           "cs_momentum", "cs_low_vol", "random")
               if experiment_id(m, h) is not None]
    picked = st.multiselect("Modellek", options, default=[m for m in ("lightgbm", "cs_reversal", "random")
                                                         if m in options],
                            format_func=label, key="cum_models")
    fig = go.Figure()
    palette = [MODEL, COMPARE, THIRD, "#eda100", "#e87ba4"]
    for i, model in enumerate(picked[:5]):
        ic = daily_ic(experiment_id(model, h))
        fig.add_scatter(x=ic.index, y=ic.cumsum(), name=label(model), mode="lines",
                        line=dict(color=palette[i], width=2),
                        hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}<extra>" + label(model) + "</extra>")
    fig.update_layout(yaxis_title="Kumulált IC", hovermode="x unified")
    chart(fig, 360)


def tab_portfolio() -> None:
    portfolio = research_table("report/models.csv", "portfolio.csv")
    if portfolio is None:
        st.info("Nincs portfólió-eredmény.")
        return
    takeaway("A felső tized vétele és az alsó tized shortolása 5 napos horizonton nyereséges "
             "költségek nélkül; 20 napon az előny jórészt eltűnik. A véletlen szabály széles sávja "
             "mutatja, mekkora a zaj ebben a mérőszámban: az IC megbízhatóbb mérce.")
    h = horizon_picker("pf_h")
    d = portfolio[(portfolio["variant"] == "final") & (portfolio["horizon"] == h)].sort_values("ls_sharpe")
    colors = [BASELINE if m in BASELINES else MODEL for m in d["model"]]
    fig = go.Figure(go.Bar(
        x=d["ls_sharpe"], y=d["model"].map(label), orientation="h", marker_color=colors,
        marker_cornerradius=4,
        error_x=dict(type="data", symmetric=False, array=d["ls_sharpe_max"] - d["ls_sharpe"],
                     arrayminus=d["ls_sharpe"] - d["ls_sharpe_min"], color=MUTED, thickness=1.2, width=4),
        hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.add_vline(x=0, line_color=AXIS, line_width=1)
    fig.update_layout(xaxis_title="Sharpe-ráta (évesített, költségek nélkül)", yaxis=dict(showgrid=False))
    chart(fig, 420)
    st.caption("A hibasáv azt mutatja, mennyit számít, melyik napon kezdjük a nem átfedő tartást "
               "(minimum és maximum az eltolások között).")
    legend_note()
    with st.expander("Táblázat"):
        view = d.sort_values("ls_sharpe", ascending=False).assign(Modell=lambda x: x["model"].map(label))
        st.dataframe(view[["Modell", "ls_ann_return", "ls_ann_vol", "ls_sharpe", "ls_max_drawdown",
                           "turnover"]], hide_index=True, width="stretch", column_config={
            "ls_ann_return": st.column_config.NumberColumn("Éves hozam", format="percent"),
            "ls_ann_vol": st.column_config.NumberColumn("Éves volatilitás", format="percent"),
            "ls_sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "ls_max_drawdown": st.column_config.NumberColumn("Max. visszaesés", format="percent"),
            "turnover": st.column_config.NumberColumn("Forgási sebesség", format="percent",
                                                      help="A portfólió cserélődő része újrasúlyozásonként.")})


def tab_absolute() -> None:
    three = research_table("final/absolute_three_way.csv", "absolute_three_way.csv")
    if three is None:
        st.info("Nincs abszolút irány eredmény.")
        return
    takeaway("Egyik irány-előrejelzés sem jobb a „mindig fel” szabálynál. A közvetlen modell "
             "megpróbálja eltalálni a piac egészének irányát, és ebben vak; az „alapráta + relatív "
             "rang” szinte mindig emelkedést mond, ezért közel azonos vele.")
    d = three.assign(Megközelítés=three["approach"].map(APPROACHES), Horizont=three["horizon"].map(lambda h: f"{h} nap"))
    fig = grouped_bars(d, "Horizont", "Megközelítés", "accuracy",
                       {APPROACHES["absolute_ensemble"]: MODEL, APPROACHES["drift_plus_rank"]: COMPARE,
                        APPROACHES["always_up"]: BASELINE}, fmt=".1%", title="Találati arány")
    fig.update_yaxes(range=[0.45, 0.58], tickformat=".0%")
    fig.update_layout(bargap=0.4)
    chart(fig, 360)
    st.caption("A függőleges tengely 45%-nál kezdődik, hogy a néhány százalékpontos különbségek látszódjanak.")
    with st.expander("Táblázat"):
        st.dataframe(d[["Horizont", "Megközelítés", "accuracy", "edge_vs_always_up", "predicted_up_share",
                        "daily_up_share_sd", "within_day_auc", "brier"]], hide_index=True, width="stretch",
                     column_config={
                         "accuracy": st.column_config.NumberColumn("Találati arány", format="percent"),
                         "edge_vs_always_up": st.column_config.NumberColumn("Különbség", format="percent"),
                         "predicted_up_share": st.column_config.NumberColumn("„Fel” jóslatok", format="percent"),
                         "daily_up_share_sd": st.column_config.NumberColumn(
                             "Napi „fel”-arány szórása", format="%.2f",
                             help="Mennyire ingadozik napról napra, hogy a jóslatok mekkora része „fel”."),
                         "within_day_auc": st.column_config.NumberColumn(
                             "Napon belüli AUC", format="%.3f", help="Egy napon belül jól rangsorol-e."),
                         "brier": st.column_config.NumberColumn("Brier", format="%.4f",
                                                                help="A valószínűségek pontossága, kisebb a jobb.")})


def tab_robustness() -> None:
    surv = research_table("survivorship/cross_sectional_by_universe.csv", "survivorship.csv")
    if surv is not None:
        st.subheader("Túlélési torzítás")
        takeaway("Ugyanaz a módszer három univerzumon. Ha a később kivezetett cégek kimaradnak, az "
                 "alacsony volatilitás szabály ereje szinte eltűnik, mert épp a zuhanó, volatilis cégek "
                 "hiányoznak. A tanuló modellek kevésbé érzékenyek; a LightGBM a helyes univerzumon a legjobb.")
        h = horizon_picker("surv_h")
        d = surv[surv["horizon"] == h].assign(Modell=lambda x: x["model"].map(label),
                                              Univerzum=lambda x: x["universe"].map(UNIVERSES))
        chart(grouped_bars(d, "Modell", "Univerzum", "ic_mean",
                           {UNIVERSES["point_in_time"]: MODEL, UNIVERSES["survivors"]: COMPARE,
                            UNIVERSES["todays_list"]: THIRD}), 360)

    runs, alt = final_runs(), final_runs("final_alt")
    if not runs.empty and not alt.empty:
        st.subheader("Alternatív adat")
        takeaway("Fundamentumok, gyorsjelentés-meglepetés és -reakció, kedvezőtlen 8-K események és "
                 "bennfentes kereskedés: ugyanaz a modell velük és nélkülük, mérhető különbség nélkül.")
        h = horizon_picker("alt_h")
        both = pd.concat([runs.assign(Változat="Ár- és piaci jellemzők"),
                          alt.assign(Változat="+ alternatív adat")])
        both = both[both["model"].isin(alt["model"]) & (both["horizon"] == h)]
        chart(grouped_bars(both.assign(Modell=both["model"].map(label)), "Modell", "Változat", "ic_mean",
                           {"Ár- és piaci jellemzők": MODEL, "+ alternatív adat": COMPARE}), 340)


STEP_LABELS = {"tree": "Fák száma", "epoch": "Epoch", "C": "C (regularizáció)"}
NO_MARKET = "_no_market"


def used_step(model: str, valid: pd.Series) -> float:
    """The setting the walk-forward actually uses."""
    if model == "logistic":
        return 0.01
    if model in ("mlp", "lstm"):
        return valid.idxmin()  # early stopping keeps the best epoch
    return valid.index.max()   # the trees always grow all of them


def tab_learning_curves() -> None:
    curves = research_table("learning_curves/curves.parquet", "learning_curves.parquet")
    if curves is None:
        st.info("Nincsenek tanulási görbék (scripts/learning_curves.py).")
        return
    takeaway("Három évben újratanítva (2012-ig, 2018-ig és 2025-ig) a modell egy félretett, későbbi "
             "időszakon mérve alig jobb a pénzfeldobásnál, és a javulás egy korai csúcs után lassan "
             "elfogy: enyhe túltanulás, nagyon kicsi jellel.")
    with st.expander("Hogyan készültek a görbék?"):
        st.markdown(
            "- A walk-forward minden évben új modellt tanít az addigi adaton. Ebből három évet "
            "választottunk: a 2012-ig, 2018-ig és 2025-ig tanított modellt.\n"
            "- Mindegyiket ugyanazokkal a beállításokkal tanítottuk újra, de a tanítóablak utolsó "
            "~10%-át (kb. az utolsó évet) félretettük validációnak, egy rés kihagyásával, hogy az "
            "átfedő címkék ne szivárogjanak át.\n"
            "- A célváltozó naponta kiegyensúlyozott (a részvények fele veri a mediánt), ezért a "
            "pénzfeldobás log-vesztesége ln 2 ≈ 0,693. A „javulás” ehhez képest mért különbség, "
            "ezredben: 1 ezred = 0,001 log-veszteség.")

    models_ = [m for m in ("lightgbm", "xgboost", "mlp", "lstm", "logistic") if m in set(curves["model"])]
    c1, c2 = st.columns([3, 2])
    model = c1.segmented_control("Modell", models_, default=models_[0], format_func=label,
                                 key="lc_model", label_visibility="collapsed") or models_[0]
    view = c2.segmented_control("Nézet", ["gain", "both", "auc"], default="gain", key="lc_view",
                                label_visibility="collapsed",
                                format_func={"gain": "Javulás", "both": "Tanító és validációs",
                                             "auc": "AUC"}.get) or "gain"
    h = horizon_picker("lc_h")

    sel = curves[(curves["model"] == model) & (curves["horizon"] == h)]
    ref = curves[(curves["model"] == "reference") & (curves["horizon"] == h)].set_index("fold_end")["value"]
    folds = sorted(sel["fold_end"].unique())
    step_name = STEP_LABELS.get(sel["step_name"].iloc[0], sel["step_name"].iloc[0])
    for col, fold in zip(st.columns(len(folds)), folds):
        f = sel[sel["fold_end"] == fold]
        fig = go.Figure()
        if view == "gain":
            for seed, g in f[(f["metric"] == "logloss") & (f["split"] == "valid")].groupby("seed"):
                v = g.set_index("step")["value"]
                gain = (ref[fold] - v) * 1000
                fig.add_scatter(x=gain.index, y=gain.values, mode="lines", showlegend=False,
                                line=dict(color=COMPARE, width=2),
                                hovertemplate="%{x}: %{y:.2f} ezred<extra></extra>")
                best = gain.idxmax()
                fig.add_scatter(x=[best], y=[gain[best]], mode="markers", showlegend=False,
                                marker=dict(size=9, color=COMPARE, line=dict(color="white", width=2)),
                                hovertemplate="legjobb: %{x}<br>%{y:.2f} ezred<extra></extra>")
                if seed == 0:
                    fig.add_vline(x=used_step(model, v), line_color=MUTED, line_width=1, line_dash="dot")
            fig.add_hline(y=0, line_color=AXIS, line_width=1)
            ytitle = "Javulás a pénzfeldobáshoz (ezred)"
        else:
            metric = "logloss" if view == "both" else "auc"
            for (split, seed), g in f[f["metric"] == metric].groupby(["split", "seed"]):
                name = "Tanító" if split == "train" else "Validációs"
                fig.add_scatter(x=g["step"], y=g["value"], mode="lines", name=name, legendgroup=name,
                                showlegend=bool(seed == 0 and fold == folds[0]),
                                line=dict(color=MODEL if split == "train" else COMPARE, width=2),
                                hovertemplate="%{x}: %{y:.4f}<extra>" + name + "</extra>")
            if metric == "logloss" and fold in ref:
                fig.add_hline(y=ref[fold], line_color=MUTED, line_width=1, line_dash="dot")
            if metric == "auc":
                fig.add_hline(y=0.5, line_color=MUTED, line_width=1, line_dash="dot")
            ytitle = "Log-veszteség" if metric == "logloss" else "AUC"
        fig.update_layout(title=dict(text=f"Tanítás {fold[:4]}-ig", font=dict(size=13, color=INK)),
                          xaxis_title=step_name, yaxis_title=ytitle if fold == folds[0] else None,
                          xaxis_type="log" if model == "logistic" else None,
                          margin=dict(l=10, r=10, t=60, b=10))
        with col:
            chart(fig, 300)
    if view == "gain":
        st.caption("Narancs: a validációs időszakon mért javulás a pénzfeldobáshoz képest (0 felett "
                   "tanult valamit); a pont a legjobb érték, a pontozott függőleges vonal a ténylegesen "
                   "használt beállítás. A neurális hálóknál minden vonal egy véletlen inicializálás.")
    else:
        st.caption("Kék: tanító adat; narancs: a félretett validációs időszak. A pontozott vízszintes "
                   "vonal a pénzfeldobás szintje.")

    if model + NO_MARKET in set(curves["model"]):
        market_ablation(curves, model, h, folds, step_name)


def market_ablation(curves: pd.DataFrame, model: str, h: int, folds: list[str], step_name: str) -> None:
    st.subheader("Piaci jellemzők nélkül")
    takeaway(MARKET_TAKEAWAY)
    sel = curves[curves["model"].isin([model, model + NO_MARKET]) & (curves["horizon"] == h)
                 & (curves["metric"] == "logloss")]
    for col, fold in zip(st.columns(len(folds)), folds):
        fig = go.Figure()
        for (variant, split), g in sel[sel["fold_end"] == fold].groupby(["model", "split"]):
            without = variant.endswith(NO_MARKET)
            name = ("Tanító" if split == "train" else "Validációs") + (" – piaci nélkül" if without else "")
            fig.add_scatter(x=g["step"], y=g["value"], mode="lines", name=name, showlegend=False,
                            line=dict(color=MODEL if split == "train" else COMPARE, width=2,
                                      dash="dot" if without else "solid"),
                            hovertemplate="%{x}: %{y:.4f}<extra>" + name + "</extra>")
        fig.update_layout(title=dict(text=f"Tanítás {fold[:4]}-ig", font=dict(size=13, color=INK)),
                          xaxis_title=step_name, yaxis_title="Log-veszteség" if fold == folds[0] else None,
                          margin=dict(l=10, r=10, t=60, b=10))
        with col:
            chart(fig, 320)
    st.caption("Kék: tanító, narancs: validációs log-veszteség. Folytonos: minden jellemzővel; pontozott: "
               "a piaci állapotot leíró oszlopok nélkül.")


pages = [
    st.Page(page_overview, title="Áttekintés", icon=":material/dashboard:", default=True),
    st.Page(page_today, title="Mai előrejelzések", icon=":material/today:", url_path="today"),
    st.Page(page_track_record, title="Éles eredmények", icon=":material/fact_check:", url_path="live"),
    st.Page(page_research, title="Kutatás", icon=":material/science:", url_path="research"),
]
st.navigation(pages, position="top").run()
