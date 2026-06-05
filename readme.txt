# MODELOVANIE TÉM V ÚLOHÁCH DETEKCIE ANTISOCIÁLNEHO SPRÁVANIA NA WEBE

Tento projekt porovnáva adaptívny a statický Random Forest pri klasifikácii textových dát v dátovom prúde. Súčasťou riešenia je aj detekcia concept driftu pomocou ADWIN a následná tematická analýza textov pomocou LDA.

## Popis projektu

Skript pracuje s článkami z databázy `nela-gt-2021.db` a so súborom `labels.csv`, ktorý mapuje zdroje na triedy. Texty sa najprv vyčistia, následne sa prevedú na embedding reprezentáciu pomocou predtrénovaného modelu GloVe a potom sa používajú na porovnanie dvoch prístupov:
- Adaptive Random Forest pre online učenie,
- statický Random Forest ako baseline.

Počas spracovania sa sledujú metriky accuracy, F1, precision a recall, a zároveň sa zaznamenávajú detegované drifty. Na vybranom drifte sa potom vykoná aj LDA analýza tém pred a po zmene.

## Požiadavky

Na spustenie projektu je potrebné mať:
- Python 3.x,
- SQLite databázu `nela-gt-2021.db`,
- súbor `labels.csv`.

Použité knižnice:
- `river`
- `gensim`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

## Inštalácia

Knižnice je možné nainštalovať príkazom:

```bash
pip install -U river gensim pandas numpy matplotlib scikit-learn
```

Ak sa skript spúšťa v notebooku, inštalácia sa vykoná automaticky v prvej bunke.

## Vstupné súbory

Projekt používa tieto vstupy:
- `nela-gt-2021.db` – SQLite databáza s článkami,
- `labels.csv` – mapovanie zdrojov na triedy.

## Spustenie

Skript je možné spustiť v Jupyter Notebooku alebo ako Python súbor.

Po spustení sa vykonajú tieto kroky:
1. načítanie databázy a labelov,
2. predspracovanie textov,
3. výber dokumentov po mesiacoch,
4. načítanie embedding modelu,
5. prevod textov na vektorové reprezentácie,
6. experiment s AdaptiveRF a StaticRF,
7. detekcia driftov pomocou ADWIN,
8. vizualizácia metrík,
9. výber vhodného driftu,
10. LDA analýza tém pred a po drifte.

## Hlavné parametre

V skripte je možné upraviť najmä tieto hodnoty:
- `MONTHCAPK` – maximálny počet dokumentov na mesiac,
- `YEARS` – roky, ktoré sa majú zahrnúť do experimentu,
- `STATIC_TRAIN_RATIO` – veľkosť tréningovej časti pre statický model,
- `DRIFT_DELTA` – citlivosť detektora ADWIN,
- `n_models` – počet stromov v Adaptive Random Forest,
- `n_estimators` – počet stromov v statickom Random Forest,
- `EMBEDDING_MODEL` – použitý predtrénovaný embedding model.

## Výstupy

Výstupom skriptu sú:
- tabuľka so zhrnutím metrík pre oba modely,
- zoznam detegovaných driftov,
- grafy vývoja accuracy, F1, precision a recall,
- porovnanie AdaptiveRF a StaticRF,
- vybraný drift vhodný na tematickú analýzu,
- LDA témy pred a po drifte,
- porovnanie tém medzi oknami pred a po zmene.

## Poznámka

Projekt je zameraný na experimentálne porovnanie modelov v časovom dátovom toku. Okomentovaný kód slúži ako technická dokumentácia systému a README ako krátka používateľská príručka.