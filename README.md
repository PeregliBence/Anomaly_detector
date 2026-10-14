# Tőzsdei előrejelző

Gépi tanuláson alapuló rendszer, amely a legforgalmasabb 1000 amerikai részvényre két dolgot jelez előre a következő 5, illetve 20 kereskedési napra:

- **relatív irány:** melyik részvény teljesít jobban a többinél (rangsor);
- **abszolút irány:** emelkedik-e a részvény ára (valószínűség és irány).

Visszamenőleg a relatív irányban a modellek a publikált stratégiáknál is jobbak voltak, az abszolút irányban viszont nem verték a „mindig felfelé” szabályt. Mindkettő élesben is kiértékelődik. A projekt BSc szakdolgozat keretében készült (BME, mérnökinformatikus szak).

A rendszer három részből áll:

- **Kutatás:** több modell és horizont kiértékelése 2008–2026-ra, túlélési torzítástól mentes adaton, purged walk-forward protokollal.
- **Éles futás:** minden hétköznap automatikusan elkészül a napi előrejelzés, és a lejárt előrejelzések kiértékelődnek. Minden előrejelzés a kimenetele előtt bekerül a repóba, így a git-történet igazolja, hogy utólag nem módosult.
- **Webes felület:** a mai előrejelzések, az éles eredmények és a kutatási eredmények.

## Felépítés

```
src/
  sources/        adatforrások: sharadar.py, yahoo.py, fred.py
  features.py     kauzális jellemzők (csak az adott napig ismert adatból)
  alt_features.py fundamentumok, gyorsjelentések, 8-K események, bennfentes kereskedés
  panel.py        panel összeállítása, keresztmetszeti jellemzők, célváltozók
  universe.py     pontbeli likvid univerzum
  models.py       baseline-ok, logisztikus regresszió, LightGBM, XGBoost
  neural.py       MLP és LSTM (PyTorch)
  validation.py   purged walk-forward, foldon belüli hiperparaméter-hangolás
  metrics.py      IC, Newey-West t-statisztika, osztályozási mérőszámok
  portfolio.py    long-short portfólió, forgás, tranzakciós költség
  store.py        kutatási adattár (SQLite)
  production.py   éles modellek tanítása, mentése, pontozás
  live.py         napi éles futás
  report.py       a dolgozat táblái
  cli.py          parancssori belépési pont
scripts/          egyedi kísérletek és diagnosztikák
tests/            automatikus tesztek (szintetikus adaton)
ui/dashboard.py   Streamlit felület
live/             éles előrejelzések, kiértékelések, univerzum
models/           a betanított éles modellek
research/         a kutatási eredmények összesítései (statisztikák, nyers adat nélkül)
```

A webes felület helyben a `data/research.db` adattárból olvas; ahol az nincs (például a Streamlit Community Cloudon), ott a `research/` mappából.

## Telepítés

Python 3.13 kell.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; Linuxon: source .venv/bin/activate
pip install -r requirements.txt
# GPU-s tanításhoz a CUDA-s torch kell:
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
cp .env.example .env              # és töltsd ki a kulcsokat
```

A verziók szándékosan rögzítettek: az éles modellek pickle-ként vannak mentve, és csak ugyanazokkal a könyvtárverziókkal tölthetők be biztonságosan.

## Használat

```bash
python scripts/sharadar_export.py           # Sharadar-pillanatkép letöltése (aktív előfizetés kell)
python scripts/final_evaluation.py          # végső kiértékelés, eredmény a kutatási adattárba
python -m src.cli report                    # a dolgozat táblái: results/report/
python -m src.cli export-research           # publikálható összesítések a research/ mappába
python -m src.cli experiments               # mi van a kutatási adattárban
python -m src.cli train                     # éles modellek tanítása a teljes történeten
python -m src.cli live-universe             # éles univerzum és jelöltlista a Sharadarból
python -m src.cli daily                     # egy napi éles futás
streamlit run ui/dashboard.py               # webes felület
python -m pytest tests                      # tesztek
```

## Napi automatikus futás

A `.github/workflows/daily.yml` minden hétköznap 22:30 UTC-kor, a New York-i zárás után lefuttatja a `python -m src.cli daily` parancsot, és commitolja a `live/` mappa változásait. Ehhez a repó beállításaiban fel kell venni a `FRED_API_KEY` titkot (Settings → Secrets and variables → Actions).

Ha a felhős futás elbukik (például a Yahoo blokkolja a GitHub szervereit), a `scripts/daily_local.ps1` helyben pótolja. A szkript fejlécében van a Windows Feladatütemezőbe való bejegyzés parancsa. Ha a felhős futás sikeres volt, a helyi futás nem csinál semmit.

Az univerzum negyedévente frissül: a `live/candidates.csv` jelöltjei közül a napi futás újrarangsorolja a 1000 legforgalmasabbat.

A modellek tanítása helyben, a Sharadar-adaton történik. Évente egyszer érdemes újratanítani őket (`python -m src.cli train`), és az új `models/` fájlokat commitolni.

## Adatok

- **Sharadar** (sharadar.com): árfolyamok a kivezetett papírokkal együtt, fundamentumok közzétételi dátummal, események, bennfentes kereskedés. Az adat licencelt, ezért **nincs a repóban**; a `data/` mappa ki van zárva a verziókezelésből.
- **Yahoo Finance:** piaci idősorok (VIX, hozamok, szektor-ETF-ek) és az éles futás árfolyamai.
- **FRED:** Moody's vállalati kötvényfelárak, egynapos késleltetéssel, mert a *t*. napi érték csak másnap jelenik meg.

## Módszertan röviden

- Minden jellemző csak az adott napig ismert adatból számolódik; ezt a `tests/test_features.py` ellenőrzi úgy, hogy levágja a jövőt, és megnézi, változik-e a múlt.
- A tanítóhalmaz minden újratanításnál a horizont hosszával a predikció előtt véget ér (purge), mert az átfedő címkék különben a tesztidőszak eredményét kódolnák.
- Az univerzumot minden újrasúlyozáskor csak az akkor elérhető adatból választom ki, a később kivezetett cégekkel együtt. A kivezetett cégek utolsó hozama az utolsó kereskedett áron zárul.
- A szignifikancia Newey-West korrekcióval számolódik; a kalibrációt véletlen pontszámokkal ellenőriztem.

## Korlátok

- A long-short eredmények tranzakciós költség nélküli, bruttó értékek. A `python -m src.cli report --costs` 5, 10 és 20 bázispontos költség melletti nettó Sharpe-rátát és forgást is közöl.
- Az éles univerzumba csak a jelöltlistán szereplő cégek kerülhetnek be.
- Az éles futás árfolyamai a Yahoo-ról, a tanításéi a Sharadarból származnak; a kettőből számolt pontszámok rangkorrelációja ugyanarra a napra 0,999.
