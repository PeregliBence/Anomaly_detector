## Téma leírása
A hallgató feladata egy olyan, reprodukálható és kutatásra alkalmas munkafolyamat kialakítása és megvalósítása, amely pénzügyi idősorokon rövid távú árfolyamirány-előrejelzést készít. A dolgozat fókusza az előrejelzési pipeline mélyebb kidolgozása, különösen a gépi tanulási modellek gyakorlati használhatóságának, megbízhatóságának és kockázat-hozam hatásának vizsgálata.

A kutatás központi kérdése, hogy az előrejelzett irányjelzések milyen pontossággal és milyen stabilitással használhatók különböző piaci környezetekben, illetve hogy a jelzésekre épített papertrading stratégia milyen teljesítményt mutat a félév során.

A hallgató feladatai:

- A felhasznált pénzügyi adatok megismerése és előkészítése, az idősorok minőségének vizsgálata (hiányzó adatok, kiugró értékek, eltérő volatilitási rezsimek).

- Egységes forecast pipeline kialakítása és implementálása az adatbetöltéstől a feature engineeringen, modellezésen és validáción át az automatikus kiértékelésig.

- Rövid távú előrejelzési feladat definiálása 5 kereskedési napos horizonton, több eszközre és azonos kiértékelési protokoll mellett.

- Modellek kialakítása és összehasonlítása:
  - baseline iránybecslés,
  - XGBoost-alapú előrejelzés,
  - LSTM-alapú előrejelzés,
  - opcionálisan hibrid vagy ensemble megközelítés.

- Walk-forward, out-of-sample validáció alkalmazása az időrendi helyesség biztosítására és az adat-szivárgás elkerülésére.

- Objektív metrikarendszer kialakítása és használata:
  - predikciós metrikák: directional accuracy, MAE, RMSE,
  - kockázat-hozam metrikák: Sharpe-ráta, max drawdown, profit factor.

- Papertrading alapú teljesítményvizsgálat elvégzése a modellek irányjelzései alapján, beleértve a historikus visszatesztet és a félév végéig történő vetített eredmények elemzését.

- Reprodukálható benchmark folyamat kialakítása fix futtatási profilokkal, rendszeres heti kiértékeléssel, valamint automatikusan mentett kutatási riportokkal.

- Kritikus összehasonlítás készítése arról, hogy az egyes modellek milyen piaci környezetekben teljesítenek stabilabban, és milyen kompromisszum van a pontosság, futásidő és általánosíthatóság között.

A téma eredményeként egy működő, gépi tanulás fókuszú előrejelzési keretrendszer jön létre, amely tudományosan védhető módon hasonlítja össze a rövid távú iránybecslési modelleket, és gyakorlati szempontból is értelmezhető teljesítményértékelést ad a féléves kutatási időszakra.
