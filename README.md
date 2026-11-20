# Unmögliche Kombinationen: was die Kurve fragt, wenn es die Zeile nie gab – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-unmoegliche-kombinationen-demo.streamlit.app/)** -->

Diese Demo gehört zur Reihe „Beitrag und Beleg“ (Modelle verstehen) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ und ist ohne die übrigen Demos der Reihe lesbar.
Sie zeigt, was die **partielle Abhängigkeit (PD)**, die **ICE-Kurve** und die **ALE** (akkumulierte lokale Effekte) fragen, wenn eine Kombination in den Daten nie vorkommt. Das Vehikel ist **erfunden**: ein Container-Generator (Verweilzeit; Art, Kunde, Zollhalt, Wochenende, Vorlauf und ein Rauschmerkmal, die Verweilzeit folgt einer bekannten Formel), in dem in der Standardeinstellung bei 45 % aller Container (Import und Leer) nie ein Vorlauf vorkommt. **Vorlauf** ist die Zeit in Tagen bis zur Schiffsabfahrt (nur Export und Umschlag haben einen); **Verweilzeit** ist die Zeit in Tagen von der Ankunft eines Containers im Hafen bis zur Abholung, so wie in `verweilzeit-demo`, aus der der Generator kopiert ist.
**Es gibt keine reale Datenquelle**; in den gelesenen Quellen gab es keine Messung, ob PD, ICE oder ALE in Betrieb, Logistik oder Recht zu besseren Entscheidungen führen. Gezählt werden Zeilen und Zufalls-Seeds; was die Zahlen für einen Betrieb heißen, wird hier nicht behauptet. Ob Menschen mit solchen Kurven besser entscheiden, zeigt diese Seite nicht; siehe die Lehrseite zu Menschenstudien.

## Kernfrage

(1) Wie viele Zeilen gibt es nicht? (2) Welche Kurve zeigt den Effekt des Vorlaufs? (3) Was zeigt die ICE auf Zeilen, die es nie gab? (4) Wie viel hängt am Anteil? (5) Fragt die ALE nie nach Unmöglichem?
**Wahrheit heißt immer „Wahrheit je Funktional“** („Funktional“ steht hier für die Rechenvorschrift der Kurve: PD, ALE oder realer Effekt; die bekannte Funktion ist μ): die PD der wahren Funktion, die ALE von μ und der **reale Effekt** (Mittel von μ über die Zeilen, in denen der Vorlauf vorkommt) sind drei verschiedene Zahlen. **Unmögliche Zeile** = eine Kombination, die der Generator nie erzeugt (Import oder Leer mit Vorlauf größer als 0, Export außerhalb von 1 bis 10 Tagen, Umschlag außerhalb von 0,5 bis 6, Zollhalt bei etwas anderem als Import); **Stütze** = Menge der möglichen Kombinationen.
Alle Anstiege sind **Anstiege der ln-Verweilzeit von Vorlauf 2 auf 5** (drei Tage), keine Steigungen je Tag.

## Befunde und Korrekturen gegenüber Plan und Vormessung

Die Zahlen der Vormessung sind Erwartungen; im Bau wurde mit dem eigenen Zufallsstrom (SplitMix64) neu gemessen, in fünf Gruppen je Zelle, und zusätzlich in einer **Zweitmessung mit anderen Zufalls-Seeds** (`data/ukb_grid_zweit.json`). Was sich verschoben hat:

* **Raster verkleinert:** Trainingsgrößen n = 300 und 3 000 (die Vormessung hatte auch 1 000), je Gruppe 6 Zufalls-Seeds (Vormessung 12), fünf Gruppen; Dosis und Fächer mit n = 1 000 und je 4 Zufalls-Seeds je Gruppe. Boosting hat 150, der Random Forest 100 Bäume.
* **Das Verhältnis „ALE näher als PD“ (eigene Wahrheit, n = 3 000)** liegt für Boosting bei 3,9 bis 9,3 (Vormessung 5,4 bis 7,2), Random Forest 3,4 bis 6,2, Netz 2,5 bis 4,1; in der Zweitmessung 4,1 bis 7,8 / 2,7 bis 5,0 / 3,3 bis 5,3: **kein fester Faktor**, die Richtung gilt bei n = 3 000 in beiden Messungen und bei allen drei trainierten Klassen; bei n = 300 nicht durchgehend (Random Forest 0,44 bis 1,5, siehe [K4]).
* **Anteil 0 (keine unmöglichen Zeilen; Bezug ist der reale Effekt, Zähleinheit: Gruppen aus je 4 Zufalls-Seeds, n = 1 000):** beim Netz ist die PD in 5 von 5 Gruppen näher am realen Effekt als die ALE (Erst- und Zweitmessung); beim Boosting ist die ALE in der Erstmessung in 3 von 5 Gruppen näher, in der Zweitmessung in 4 von 5: ein Vorteil der ALE, der für beide Klassen gilt, gibt es nicht. Die Vormessung nannte „Boosting ALE näher in 5 von 5“.
* **Anteil 0,9, Boosting:** die ALE ist schlechter als die PD in 4 von 5 Gruppen (Zweitmessung 5 von 5; Vormessung 5 von 5); beim Netz ist die ALE besser in 4 von 5 (Zweitmessung 5 von 5).
* **Netz, flache unmögliche Kurven:** 6 bis 30 % (n = 3 000; Zweitmessung 12 bis 26 %; Vormessung 11 bis 22 %).
* **Boosting, Anstieg in der PD** 0,68 bis 0,75 (Vormessung 0,69 bis 0,72), jetzt mit der Zweitmessung 0,65 bis 0,73.

## Modell

Der Vorlauf ist bei Import und Leer immer 0 (Anteile 0,35 und 0,10), bei Export gleichverteilt über [1, 10], bei Umschlag über [0,5; 6]; der Median der Verweilzeit ist bei Export 0,6 + 0,85 v Tage, bei Umschlag 0,3 + 0,8 v Tage (die Log-Mitte μ ist sein Logarithmus plus Kundeneffekt), bei Import und Leer unabhängig von v. Der Generator ist aus `verweilzeit-demo` **kopiert** (`ukb_generator.py`, Kopie-Test gegen eingefrorene Werte der Quelle), mit dem Anteil ohne Vorlauf als Regler (Standard 0,45) und mit SplitMix64 gezogen. **Konvention:** Betragsmaß ist die mittlere Abweichung der Kurve über die Punkte 3 bis 6 (relativ zu 2) und der Anstieg 2 auf 5; **Rangmaß der Verfahren** ist der Anteil der Zufalls-Seeds, in denen die ALE näher an der Bezugskurve liegt als die PD (ein Rangmaß der Kurven wäre trivial, alle Kurven sind monoton).

## Methodik

* **PD** (eigene Implementierung): den Vorlauf für **alle** Zeilen auf den Gitterwert setzen und mitteln. **ICE:** dieselbe Kurve je Zeile; die PD ist ihr Mittel. **ALE:** Fächer aus Quantilen (K = 20 angefordert; wegen der vielen Nullen bleiben 11 bis 12 Fächer, siehe [K8]), je Fach die Differenzen an den Fachgrenzen mitteln, aufsummieren, zentrieren; das erste Fach enthält alle Zeilen mit Vorlauf 0. **Wahrheit:** PD und realer Effekt der bekannten Funktion in geschlossener Form, ALE als Integral der bedingten Ableitung (Gauß-Legendre).
* **Modellklassen** (scikit-learn, eingefroren): Boosting (150 Bäume, Tiefe 3), Random Forest (100 Bäume), Netz (16 Einheiten, auf R² gegen μ abgestimmt), lineares Modell; Hyperparameter fest. **Zähleinheiten:** Zeilen, Zufalls-Seeds (je Zufalls-Seed ein Trainingssatz und 3 000 frische Auswertungszeilen; ICE-Anstiege und flache Kurven an den ersten 2 000 davon, siehe [K6]), Auswertungspunkte der ALE (Kanten der Fächer), Gruppen (6 bzw. 4 getrennte Zufalls-Seeds); Aussagen über Verhältnisse und Anteile sind **Spannen über fünf Gruppen**. Live in der App: exakte Anteile, die Wahrheitskurven, das lineare Modell (kleinste Quadrate); eingefroren: Boosting, Random Forest, Netz und alle Gruppenmittel (`tools/ukb_make_grid.py`, `--check`, `--zweit`).

## Befunde (gemessen, keine Behauptungen)

Die Kennung in eckigen Klammern verweist auf die Tabelle „Zahl → Test“.

**1. Anteil unmöglicher Zeilen** (Zähleinheit: Zeilen; exakt in Brüchen). [K1] Der Anteil der Container ohne Vorlauf ist **exakt 0,45** (0,35 + 0,10). Setzt man ein Merkmal für **alle** Zeilen: Vorlauf auf 2 bis 6 Tage 9/20 = 0,45; Art Import 11/20 = 0,55; Art Export 26/55 = 0,4727; Art Umschlag 7/12 = 0,5833; Art Leer 51/80 = 0,6375; Zollhalt 1: 13/20 = 0,65 (gegen Aufzählung der Zellen und eine Stichprobe von 8 000 Zeilen im Band).

**2. PD, real und ALE der wahren Funktion** (geschlossen). [K2] Anstieg von Vorlauf 2 auf 5: PD 0,30·ln(4,85/2,3) + 0,25·ln(4,3/1,9) = **0,42801**, realer Effekt 0,42801/0,55 = **0,77820**, ALE (Integral der bedingten Ableitung) **0,78685**; PD geteilt durch real ist exakt der Anteil möglicher Zeilen 0,55 (allgemein 1 − Anteil); ALE minus real 0,0087. Die Mini-Welt von Hand (drei Zellen): PD 8/5 gegen real 4, Modell-PD 14/5, ALE-Anstieg 4 für beide, erstes Fach 13/11 gegen 4/11.

**3. Vier Modellklassen, drei Bezugskurven** (Zähleinheit: Zufalls-Seeds; n = 3 000; Spanne über fünf Gruppen zu je 6 Zufalls-Seeds). [K3] Anstieg 2 auf 5 in der PD / in der ALE: Boosting 0,68 bis 0,75 / 0,76 bis 0,80, Random Forest 0,60 bis 0,63 / 0,75 bis 0,79, Netz 0,53 bis 0,62 / 0,81 bis 0,84, lineares Modell 0,64 bis 0,65 / 0,64 bis 0,65; die Modell-PD liegt in **allen 40 Gruppenmitteln** (4 Klassen, n = 300 und 3 000, 5 Gruppen) zwischen der PD der Wahrheit (0,43) und dem realen Effekt (0,78). [K4] ALE näher als PD **gegen die jeweils eigene Wahrheit** (Verhältnis der Abweichungen): Boosting 3,9 bis 9,3, Random Forest 3,4 bis 6,2, Netz 2,5 bis 4,1, lineares Modell 1,3 bis 1,4; **gegen den realen Effekt** 1,4 bis 3,7 / 2,5 bis 4,1 / 2,8 bis 5,4 / 1,0 (Zweitmessung 1,6 bis 2,4 / 2,4 bis 4,7 / 3,5 bis 5,0 / 1,0); bei n = 300 gegen die eigene Wahrheit 1,3 bis 3,7 / 0,44 bis 1,5 / 0,91 bis 1,2 / 1,0 bis 1,5 (beim Random Forest und Netz ist die PD in einzelnen Gruppen näher). **PD über die Stütze** (nur über die möglichen Zeilen gemittelt; setzt voraus, dass die Stütze bekannt ist), Abweichung vom realen Effekt gegen die der ALE (n = 3 000, Spanne über fünf Gruppen, Erstmessung / Zweitmessung): Netz 0,027 bis 0,041 gegen 0,035 bis 0,055 / 0,029 bis 0,037 gegen 0,039 bis 0,049 (Stütze näher in 5 von 5 / 5 von 5 Gruppen); Boosting 0,057 bis 0,063 gegen 0,026 bis 0,055 / 0,033 bis 0,050 gegen 0,027 bis 0,051 (Stütze näher in 0 von 5 / 1 von 5); Random Forest 0,036 bis 0,064 gegen 0,033 bis 0,047 / 0,031 bis 0,050 gegen 0,028 bis 0,053 (Stütze näher in 1 von 5 / 3 von 5): beim Netz liegt die Stütze näher, beim Boosting die ALE, beim Random Forest wechselt es. [K5] Anteil der unmöglichen Auswertungspunkte der ALE (K = 20 angefordert, tatsächlich 11 bis 12 Fächer, siehe [K8]): 24,2 bis 25,1 % (n = 3 000); der Anteil hängt nur von den Auswertungszeilen und K ab, nicht vom Modell, und ist deshalb für alle vier Klassen derselbe. Beim linearen Modell sind PD, ALE und die PD über die Stütze **gleich** (live gerechnet mit einem Zufalls-Seed: 0,665, gegen die Gruppenmittel 0,64 bis 0,65 aus [K3]; alle ICE-Kurven derselbe Anstieg, je Tag 0,222 in ln-Einheiten, also ein fester Prozentzuschlag auf die Verweilzeit, kein fester Betrag in Tagen).

**4. ICE auf Zeilen, die es nie gab** (Import und Leer; Wahrheit exakt 0; n = 3 000; Anstieg und flache Kurven an den ersten 2 000 der 3 000 Auswertungszeilen je Zufalls-Seed, der Fehler bei Vorlauf 5 an allen 3 000; Fehler bei Vorlauf 5 = mittlerer Betrag der Abweichung des Modells von μ, wenn der Vorlauf für alle Zeilen auf 5 gesetzt wird). [K6] Mittlerer Betrag des Anstiegs 2 auf 5: Boosting 0,57 bis 0,64, Random Forest 0,42 bis 0,45, Netz 0,21 bis 0,39, lineares Modell 0,64 bis 0,65; flache Kurven (Anstieg unter 0,1): 0,0 % / 0,1 bis 0,3 % / 6 bis 30 % / 0 %; der Fehler bei Vorlauf 5 ist auf unmöglichen Zeilen 4,0 bis 6,5 / 3,1 bis 3,8 / 9,4 bis 15,3 / 6,6 bis 7,0 mal so groß wie auf möglichen, bei R² gegen μ auf natürlichen Zeilen von 0,955 bis 0,959 / 0,895 bis 0,902 / 0,961 bis 0,965 / 0,888 bis 0,889: **der Testfehler auf natürlichen Zeilen sieht das nicht.**

**5. Dosis: der Anteil ohne Vorlauf** (n = 1 000, 6 000 Auswertungszeilen, fünf Gruppen zu je 4 Zufalls-Seeds). [K7] PD-Anstieg der Wahrheit bei Anteil 0 / 0,2 / 0,45 / 0,7 / 0,9 (geschlossene Form, (1 − Anteil) · 0,778): 0,778 / 0,623 / 0,428 / 0,233 / 0,078 gegen real 0,778 (Verhältnis 1 − Anteil); die eingefrorene Stichprobe der 6 000 Auswertungszeilen gibt bei 0,7 den Wert 0,235, den die Tabelle der App zeigt. Abweichung der Modellkurve vom realen Effekt, PD gegen ALE: **Anteil 0** Boosting 0,05 bis 0,13 gegen 0,03 bis 0,07 (ALE näher in 3 von 5 Gruppen), Netz 0,03 bis 0,06 gegen 0,04 bis 0,07 (PD näher in 5 von 5): **kein Vorteil der ALE, der für beide Klassen gilt**; **Anteil 0,9** Boosting 0,15 bis 0,21 gegen 0,11 bis 0,48 (ALE schlechter in 4 von 5 Gruppen), Netz 0,31 bis 0,44 gegen 0,16 bis 0,43 (ALE besser in 4 von 5); bei 0,7 liegt die PD des Netzes 0,28 bis 0,34 daneben, die ALE 0,02 bis 0,06.

**6. ALE ist nicht gratis** (n = 1 000, fünf Gruppen zu je 4 Zufalls-Seeds; Zahl der angeforderten Fächer K). [K8] Anteil unmöglicher Auswertungspunkte der ALE bei K = 5 / 10 / 20 / 50: 0,318 bis 0,320 / 0,268 bis 0,270 / 0,240 bis 0,251 / 0,231 bis 0,238. **K ist die angeforderte Zahl:** bei 45 % Zeilen ohne Vorlauf fallen die unteren Quantilkanten bei 0 zusammen (`np.unique`), tatsächlich bleiben bei K = 5 / 10 / 20 / 50 nur 3 / 6 / 11 bis 12 / 27 bis 30 Fächer (3 000 Auswertungszeilen je Zufalls-Seed, 20 Zufalls-Seeds je Messung, beide Messungen), und bei K = 20 stammen 88 bis 99 % der unmöglichen Auswertungspunkte aus dem ersten Fach. Im ersten Fach (Vorlauf 0 wird an eine obere Kante größer als 0 geschoben) steht beim Boosting bei K = 20 ein Effekt von −0,91 bis −0,65 statt +0,04 bis +0,09 bei der wahren Funktion; die Abweichung der ALE gegen den realen Effekt ist beim Boosting bei K = 5 0,30 bis 0,36, bei K = 10 0,04 bis 0,05 (beim Boosting ist K = 10 in allen 10 Gruppen von Erst- und Zweitmessung besser als K = 20; die Reihung von K = 20 und 50 und die des Netzes sind nicht stabil, der Sprung bei K = 5 schon); bei K = 5 reicht das erste Fach bis Vorlauf 2,52 bis 2,67, ab K = 10 liegt seine obere Kante unter 2.

## Zahl → Test

| Kennung | Zahl (Kurzform) | Test (Band oder exakt) |
|---|---|---|
| K1 | Anteile 0,45, 9/20, 11/20, 26/55, 7/12, 51/80, 13/20 | `test_handbeispiele.py` (exakt, Brüche, Aufzählung und Stichprobe im Band) |
| K2 | 0,42801 / 0,77820 / 0,78685, 0,55, 0,0087, Mini-Welt | `test_handbeispiele.py` (exakt bzw. 5e-6), `test_oracle_kurven.py` (Integral mit scipy) |
| K3 | Anstiege PD und ALE je Klasse, 40 Gruppenmittel | `test_claims.py::test_k3_*` (Band 0,006 auf zwei Stellen) |
| K4 | Verhältnisse ALE gegen PD (auch Zweitmessung gegen den realen Effekt), Spannen, PD über die Stütze gegen ALE | `test_claims.py::test_k4_*` (Band, Zählung je Gruppe) |
| K5 | 24,2 bis 25,1 %, lineares Modell live | `test_claims.py::test_k5_*`, `test_oracle_kurven.py` |
| K6 | ICE auf unmöglichen Zeilen: Anstieg, flach, Fehlerverhältnis, R² | `test_claims.py::test_k6_*` (Band) |
| K7 | Dosis: Wahrheit exakt, Abweichungen, Gruppenzählungen | `test_claims.py::test_k7_*`, `test_handbeispiele.py` (exakt) |
| K8 | Fächer K: unmögliche Punkte, erstes Fach, Abweichung, tatsächliche Fächerzahl und Anteil des ersten Fachs (neu gerechnet aus dem Generator) | `test_claims.py::test_k8_*` (Band bzw. exakte Zählung) |
| Z | Zweitmessung (Befunde und Korrekturen) | `test_claims.py::test_zweitmessung_*` |

Messungen dieser README, die **nicht** im Test stehen: die Vergleichswerte der Vormessung in Klammern unter „Befunde und Korrekturen“ (Zahlen der Planung, nicht dieser Demo).

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Offen |
|---|---|---|
| **Harte strukturelle Abhängigkeit** | Der Vorlauf ist bei Import und Leer exakt 0 (Nullen, ein Träger mit Lücken). Weiche Abhängigkeit (stetige Korrelation) ist hier nicht gemessen. | weiche Abhängigkeit |
| **Regression, ein Merkmal** | Gemessen ist die Wirkung des Vorlaufs auf die Log-Verweilzeit; keine Klassifikation, keine kategorialen Merkmale (ALE für Art, Kunde), keine zweidimensionale ALE oder PD zweier Merkmale. | Klassifikation, kategoriale und zweidimensionale Kurven |
| **Vier Modellklassen mit festen Einstellungen** | Boosting (150 Bäume), Random Forest (100 Bäume) und das lineare Modell sind nicht abgestimmt, das Netz (16 Einheiten) ist auf R² gegen μ abgestimmt; die Einstellungen sind nicht variiert, scikit-learn-Modelle sind eingefroren. | andere Einstellungen |
| **Kurvenabweichung und Anstieg** | Betragsmaß ist die mittlere Abweichung der Kurve über die Punkte 3 bis 6 und der Anstieg von Vorlauf 2 auf 5; ein Rangmaß der Kurven wäre trivial (alle Kurven sind monoton). | Rangmaße der Wichtigkeit: andere Demo |
| **Ursache nicht untersucht** | Warum ein Modell auf unmöglichen Zeilen einen Effekt erfindet (Baumform, Extrapolationsregel), ist nicht untersucht; Deutungen stehen als Deutung da. | Ursache |
| **Zählungen aus Zufalls-Seeds, eine Plattform** | Aussagen als Spanne über Gruppen. Gemessen unter Windows mit Python 3.14; Linux-Lauf (Docker, Python 3.12) am 2026-11-20 grün. | - |
| **Nutzen von PD, ICE und ALE** | In den gelesenen Quellen gab es keine Messung, ob PD, ICE oder ALE in Betrieb, Logistik oder Recht zu besseren Entscheidungen führen; hier wird nichts dazu behauptet. Ob Menschen mit solchen Kurven besser entscheiden, zeigt diese Seite nicht; siehe die Lehrseite zu Menschenstudien. | nicht belegt |

## Bewusst nicht enthalten / nicht gemessen

* 🚧 **Klassifikation, kategoriale ALE (Art, Kunde), zweidimensionale ALE und PD zweier Merkmale** sind nicht gemessen. 🚧 **Weiche Abhängigkeit** (Zwillinge mit Korrelation 0,9) ist `korrelierte-merkmale-demo` (dort Permutation, nicht PD), PD, ICE und ALE bei stetiger Korrelation in diesem Vehikel nicht gemessen.
* 🚧 **Netz:** auf R² gegen μ abgestimmt, ein zweites Netz mit schwächerer Regularisierung ist nicht gemessen; Boosting und Random Forest mit festen Standardwerten. 🚧 **Ursache** des erfundenen Effekts nicht untersucht. 🚧 **Plattform:** gemessen unter Windows, Python 3.14; Linux-Lauf (Docker, Python 3.12) am 2026-11-20 grün. 🚧 **Wirkung in Betrieb:** nicht belegt.
* **Abgrenzung:** `korrelierte-merkmale-demo` zeigt, dass das Mischen eines Zwillings Zeilen erzeugt, die es nie gab, für **Permutation** und **stetige** Korrelation; hier sind es **PD, ICE, ALE**, eine **harte** Abhängigkeit (exakter Bruch je Merkmal) und die **drei Wahrheiten** der Kurve. Kein Shapley-Wert hier.

## Tests

Handbeispiele in Brüchen (`test_handbeispiele.py`), Orakel mit anderem Rechenweg (`test_oracle_kurven.py`), Daten und Claims dieser README (`test_claims.py`), Kopie des Generators, Verdikte, Randfälle, Presets, Layout, Footer, App (AppTest), Laufzeit-Importe, Texte. Die Suite hat 95 Tests; sie lief lokal mit `-n 4` (Windows, Python 3.14, Maschine unter Last durch parallele Arbeiten) in rund 11 Sekunden; die AppTest-Läufe überwiegen. Das Neu-Rechnen eines Zufalls-Seeds (`test_make_tools.py`) vergleicht als Band.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml`, `.gitignore`, `.streamlit/config.toml` | CI (Push, Pull Request, wöchentlicher Canary), Ignorierliste, Streamlit-Einstellungen |
| `app.py` | Streamlit-App: Kernfragen, fünf Experimente, Grenzen, Formeln |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit (streamlit, numpy, plotly) bzw. plus pytest, scikit-learn, scipy für Tests und Werkzeuge |
| `data/ukb_grid.json`, `data/ukb_grid_zweit.json` | eingefrorene Messungen mit Quelle und Stand |
| `tools/ukb_make_grid.py` | Raster neu erzeugen und prüfen (`--check`, `--zweit`; Dev-venv) |
| `ukb_generator.py`, `ukb_feasible.py` | kopierter Generator, Gültigkeit und exakte Anteile |
| `ukb_curves.py`, `ukb_models.py` | PD, ICE, ALE und Wahrheitskurven; lineares Modell (kleinste Quadrate) |
| `ukb_data.py`, `ukb_experiments.py`, `ukb_evaluation.py` | Laden, Rechnungen der Experimente, Verdikte |
| `ukb_rng.py` | SplitMix64 (Vorlage aus dem Website-Repo) |
| `ukb_constants.py`, `ukb_presets.py`, `ukb_layout.py`, `ukb_visualization.py`, `ukb_texts.py` | Regler, Presets mit Permalink, Kopf, Abbildungen, Texte |
| `tests/` | siehe Abschnitt Tests |

## Quellen

Stufe W = im Wortlaut gelesen. **W:** Apley, Zhu 2020 (ALE): Gleichungen 1, 2 und 6, Satz 1, Bemerkung 1 (S. 6), Beispiele 1 und 2, Abschnitt zu den Rechenkosten; Goldstein u. a. 2015 (ICE), S. 4 und 5. **Nicht gelesen:** Friedman 2001 (Scan ohne Textebene; die PD-Definition steht in den beiden gelesenen Arbeiten), Hooker, Mentch, Zhou (nur über Kumar u. a. referiert), Molnar (Zusammenfassung). Vorbilder: XAI-Bench, OpenXAI, Hooker u. a.; neu ist keine Aussage, nur die nachrechenbare Vorführung.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Das Raster neu erzeugen oder prüfen (Dev-venv, einige Minuten bis eine Viertelstunde): `python tools/ukb_make_grid.py --check`; `--zweit` rechnet mit anderen Zufalls-Seeds.

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
