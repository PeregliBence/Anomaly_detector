## Téma leírása
A hallgató feladata egy olyan, reprodukálható munkafolyamat kialakítása és megvalósítása, amely pénzügyi idősorokon hatékonyan képes anomáliák detektálására, valamint rövid távú árfolyamirány-előrejelzések készítésére. A rendszer célja, hogy kutatási környezetben stabilan és elfogadható futási idővel működjön, az eredmények pedig objektíven összehasonlíthatók és dokumentálhatók legyenek.

A téma fókusza a gyakorlati használhatóság és a védhető kutatási módszertan:
- skálázható és jól strukturált scope-ok,
- egy stabil alapmodell-csomag,
- egységes értékelési protokoll,
- féléves rendszeres benchmarkolás.

A hallgató feladatai:

- A felhasznált adatforrások és idősorok sajátosságainak megismerése (adatminőség, hiányzó értékek, volatilitási rezsimek).

- Egy egységes, lépésről lépésre követhető feldolgozási pipeline kialakítása az adatbetöltéstől a feature engineeringen és modellezésen át a kiértékelésig.

- Anomáliadetektálási alaprendszer kidolgozása Rolling Z-score és Isolation Forest módszerekkel, majd ezek ensemble jellegű összesítése súlyossági besorolással.

- Rövid távú árfolyamirány-előrejelzési rendszer kialakítása XGBoost, illetve ARIMA+XGBoost modellkombináció összehasonlításával.

- Walk-forward, out-of-sample validációs eljárás alkalmazása, amely biztosítja az időrendi helyességet és csökkenti az adat-szivárgás kockázatát.

- Objektív metrikarendszer kialakítása:
  - anomáliákra: precision, recall, F1;
  - előrejelzésre: directional accuracy, MAE/RMSE;
  - papertrading értelmezésre: Sharpe-ráta, max drawdown, profit factor.

- Olyan papertrading-alapú értékelési eljárás kialakítása, amely a modelljelzésekből származó historikus és félév végéig vetített hozamprofilt is megadja.

- Heti, reprodukálható benchmark futtatás bevezetése fix profil alapján, dátumozott eredménymappákkal és automatikus kutatási riportokkal.

- Az eredmények kritikus elemzése különböző piaci környezetekben, különösen annak vizsgálata, hogy mely feltételek mellett marad stabil a rendszer teljesítménye.

- Kiterjesztési lehetőségek bemutatása: LSTM-alapú megközelítések vizsgálata kísérleti kiegészítőként, az alaprendszerhez viszonyítva.

A téma eredményeként egy stabil, kutatásra kész döntéstámogató keretrendszer jön létre, amely egyszerre szolgálja az anomáliadetektálás és az árfolyamirány-előrejelzés tudományos értékelését, valamint a szakdolgozati következtetések megbízható alátámasztását.
