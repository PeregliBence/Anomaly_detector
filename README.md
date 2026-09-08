# 📈 Tőzsdei Anomáliadetektálás és Előrejelzés

Professzionális tőzsdei anomáliadetektáló és előrejelző rendszer, amely adatvezérelt módszerekkel elemzi a pénzügyi idősorokat.

## Funkciók

### Anomáliadetektálás (10 módszer)
- **Z-score**: Standard, robosztus és gördülő ablakos változatok
- **IQR**: Interkvartilis terjedelem alapú outlier detektálás
- **Isolation Forest**: Multi-run ensemble a stabilabb eredményekért
- **LOF (Local Outlier Factor)**: Lokális sűrűségkülönbség alapú
- **One-Class SVM**: RBF kernel nemlineáris boundary-val
- **LSTM Autoencoder**: Szekvenciális mintázatok rekonstrukciós hibájából detektál
- **STL dekompozíció**: Szezonális felbontás reziduáljainak elemzése
- **CUSUM change-point**: Kumulatív összeg monitoring regime váltásokhoz
- **Volume anomália**: Forgalmi csúcspontok és intézményi kereskedés jelzése
- **Volatilitás rezsimváltás**: Piaci stressz korai jelzése

### Előrejelzés (valószínűségi kimenettel)
- **ARIMA**: Statisztikai idősoros modell konfidencia intervallummal
- **XGBoost**: Gradient boosted gépi tanulás bootstrap bizonytalansággal
- **LSTM**: Neurális háló MC dropout alapú konfidenciával
- **Ensemble**: Confidence-súlyozott kombináció

### Egyéb
- 18+ technikai feature (RSI, MACD, Bollinger, ATR, momentum, drawdown, stb.)
- Trend és volatilitási rezsim klasszifikáció
- Anomália súlyossági szintek (mild/strong)
- Feature fontosság elemzés
- Adatminőségi ellenőrzés
- CSV export
- Live mód automatikus frissítéssel
- KPI dashboard

## Telepítés

```bash
# 1. Virtuális környezet létrehozása
python -m venv python_env

# 2. Aktiválás
python_env\Scripts\activate    # Windows
source python_env/bin/activate # Linux/Mac

# 3. Csomagok telepítése
pip install -r requirements.txt
```

## Indítás

### Streamlit dashboard (vizuális felület)
```bash
streamlit run streamlit_app.py

# Vagy Windows-on a batch file-lal
run_app.bat
```

### Szakdolgozati kutatási futtatás (automatikus kiertes)

Az alkalmazas tartalmaz egy CLI kutatasi futtatot, ami egyszerre ad:
- anomalia detektalas minosegmutatokat (precision, recall, F1)
- elorejelzesi iranytalalati mutatokat (directional accuracy, Sharpe, drawdown)
- papertrading jellegu hozam becslest a felev vegig

```bash
# Alap futtatas
python thesis_runner.py

# Profil alapjan (ajanlott, reprodukalhato)
python thesis_runner.py --profile thesis_profile.json

# Egyedi ticker lista es idoszak
python thesis_runner.py --tickers AAPL,MSFT,^GSPC,BTC-USD --start 2020-01-01 --semester-end 2026-12-31

# Windows batch indito
run_thesis_research.bat

# Heti, datumozott benchmark futtatas
run_thesis_weekly.bat
```

Kimeneti fajlok a `results/thesis/` mappaba kerulnek:
- `forecast_benchmark.csv`
- `anomaly_benchmark.csv`
- `selected_models.csv`
- `deployment_decisions.md`
- `research_summary.md`
- `run_details.json`

Az uj, gyakorlatias pipeline minden tickerre automatikusan ad:
- benchmark referencia sorokat (`benchmark_always_long`, `benchmark_always_short`, `benchmark_random_50`)
- egy GO/NO-GO dontest az adott futas legjobb modellkonfiguraciojara
- indoklast, ha a modell nem eri el a gate-eket vagy nincs edge az always-long baseline-hoz kepest

Heti futtatasnal a kimenet datum szerinti mappakba kerul:
- `results/thesis_runs/YYYY-MM-DD/`

Reszletes, féléves hasznalati folyamat:
- `THESIS_GUIDE.md`

### Stable Research Mode (ajanlott alap)

Az alap profil kifejezetten szakdolgozati, stabil es gyors futasra van hangolva:
- Forecast modellek: `xgboost`, `arima+xgboost`
- Fix horizont: 5 nap
- Heti benchmark futas datumozott mappakba

```bash
# Alap, reprodukalhato futas
python thesis_runner.py --profile thesis_profile.json

# Heti rutin futtatas (Windows)
run_thesis_weekly.bat
```

Megjegyzes (Z-score vs LSTM):
- A stabil alapban Z-score marad az egyik fo anomalia-jelzo (gyors, ertelmezheto, robusztus).
- LSTM-et erdemes opcionális, kiserleti kiegesziteskent futtatni, nem alapmodellkent.

### ML Track (konzulensi ML fokuszhoz)

Kulon profil a modellek osszehasonlitasahoz:
- `xgboost`
- `arima+xgboost`
- `arima+xgboost+lstm`

```bash
# ML profil futtatasa
python thesis_runner.py --profile thesis_profile_ml.json

# Heti ML benchmark (Windows)
run_thesis_ml_weekly.bat
```

ML kimenetek:
- `results/thesis_ml/` (kezifuttatas)
- `results/thesis_ml_runs/YYYY-MM-DD/` (heti datumozott futtatas)

### LSTM-heavy Track (kiserleti)

Kulon profil, szukebb tickerlistaval a futasido kordaban tartasahoz:
- `xgboost`
- `arima+xgboost+lstm`
- `lstm`

```bash
# LSTM profil futtatasa
python thesis_runner.py --profile thesis_profile_lstm.json

# Heti LSTM benchmark (Windows)
run_thesis_lstm_weekly.bat
```

LSTM kimenetek:
- `results/thesis_lstm/`
- `results/thesis_lstm_runs/YYYY-MM-DD/`

### Baseline vs ML osszehasonlitas

```bash
python compare_forecast_runs.py --baseline-dir results/thesis --candidate-dir results/thesis_ml --label-baseline baseline --label-candidate ml --out-dir results/comparisons/baseline_vs_ml

# Windows batch
run_compare_baseline_ml.bat
```

### Egyszeru forecast-only evaluator (ajanlott, ha csak heti becsles vs teny kell)

Ez a mod pontosan azt csinalja, amit a minimal workflow igenyel:
- historikus rolling train
- kovetkezo 5 nap predikcio
- predikalt irany osszevetese a tenyleges irannyal
- window-szintu CSV + egyszeru osszefoglalo

```bash
# CLI
python simple_forecast_runner.py --profile simple_forecast_profile.json

# Windows one-click
run_simple_forecast.bat

# Windows weekly dated output
run_simple_forecast_weekly.bat
```

Kimenetek:
- `results/simple_forecast/forecast_summary.csv`
- `results/simple_forecast/forecast_windows.csv`
- `results/simple_forecast/simple_summary.md`
- `results/simple_forecast/run_details.json`

Heti datumozott kimenet:
- `results/simple_forecast_runs/YYYY-MM-DD/`

### Egygombos teljes heti futtatás

```bash
run_all_thesis_workflows.bat
```

Ez lefuttatja sorban:
- baseline profil
- ML profil
- LSTM profil
- baseline vs ML es baseline vs LSTM osszehasonlitast

Hasznos CLI opciok a forecast kutatashoz:
- `--focus forecast-only` vagy `--focus full`
- `--min-confidence 0.55` (trade szures)
- `--fee-bps 2 --slippage-bps 3` (papertrading koltsegmodell)
- `--benchmark-seed 42` (reprodukálható random benchmark)
- `--selection-min-trades 15`
- `--selection-min-sharpe 0.2`
- `--selection-max-drawdown-pct 35`
- `--selection-min-diracc 0.50`

## Projektstruktúra

```
app/                      # Fő alkalmazáscsomag
├── __init__.py
├── config.py             # Konfiguráció (.env támogatás)
├── data_loader.py        # Yahoo Finance adatletöltés
├── features.py           # Feature engineering (18+ feature)
├── anomaly_models.py     # 10 anomáliadetektálási módszer
├── forecasting.py        # ARIMA / XGBoost / LSTM előrejelzés
├── ensemble.py           # Ensemble scoring és súlyossági szintek
├── visualization.py      # Plotly interaktív grafikonok
├── explainability.py     # Magyarázat generálás
└── search.py             # Ticker keresés

streamlit_app.py          # Streamlit főalkalmazás
tests/                    # Tesztek
├── test_new_modules.py   # Új modulok tesztjei
├── test_anomaly_models.py # Legacy tesztek
└── run_quick_tests.py

requirements.txt          # Python csomagok
.env.example              # Konfigurációs minta
run_app.bat               # Windows indító script
thesis_runner.py          # Szakdolgozati kutatasi evaluator (CLI)
run_thesis_research.bat   # Windows one-click kutatasi futtatas
run_thesis_weekly.bat     # Heti, datumozott benchmark futtatas
run_thesis_ml_weekly.bat  # Heti ML benchmark futtatas
run_thesis_lstm_weekly.bat # Heti LSTM benchmark futtatas
run_compare_baseline_ml.bat # Baseline vs ML osszehasonlito futtatas
run_all_thesis_workflows.bat # Teljes heti master futtatas
thesis_profile.json       # Reprodukálható kutatasi profil
thesis_profile_ml.json    # ML track profil
thesis_profile_lstm.json  # LSTM-heavy track profil
compare_forecast_runs.py  # Benchmark osszehasonlito riport generator
THESIS_GUIDE.md           # Féléves szakdolgozati hasznalati utmutato
WEEKLY_CHECKLIST.md       # Heti kutatasi naplo/checklist sablon
MODEL_SELECTION_POLICY.md # Elore rogzitett modellvalasztasi szabalyok
```

## Konfiguráció

Másold a `.env.example` fájlt `.env` néven és módosítsd igény szerint:

```bash
cp .env.example .env
```

Fontosabb beállítások:
- `DEFAULT_TICKER`: Alapértelmezett eszköz (pl. `AAPL`)
- `ZSCORE_THRESHOLD`: Z-score érzékenység (2.0-5.0)
- `IF_CONTAMINATION`: Isolation Forest contamination (0.01-0.10)
- `ENSEMBLE_VOTE_K`: Hány modellnek kell egyetértenie (1-5)
- `FORECAST_HORIZON`: Előrejelzési horizont napokban

## Támogatott eszközök

| Típus | Példák |
|-------|--------|
| 🏢 Részvények | AAPL, MSFT, TSLA, NVDA |
| 📊 Indexek | ^GSPC (S&P 500), ^IXIC (NASDAQ), ^GDAXI (DAX), ^BUX |
| 📦 ETF-ek | SPY, QQQ, GLD, VTI |
| 💱 Devizapárok | EURUSD=X, EURHUF=X |
| ₿ Kriptovaluták | BTC-USD, ETH-USD, SOL-USD |

## Technológiák

- **Frontend**: Streamlit
- **Vizualizáció**: Plotly
- **ML**: TensorFlow/Keras, scikit-learn, XGBoost
- **Idősoros elemzés**: statsmodels
- **Adatforrás**: Yahoo Finance (yfinance)

## Fontos megjegyzés

⚠️ Ez az alkalmazás kizárólag oktatási és kutatási célokra készült. **Nem minősül befektetési tanácsadásnak.** Az előrejelzések adatvezérelt becslések, nem garanciák.
