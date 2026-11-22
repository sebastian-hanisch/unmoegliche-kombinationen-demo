"""Anzeigetexte der App (und die Textstücke, die die README wörtlich wiederholt: `tests/test_texts.py` vergleicht beide). Alles Erfundene ist als erfunden gekennzeichnet; es gibt keine reale Datenquelle."""

TITEL = "Unmögliche Kombinationen: was die Kurve fragt, wenn es die Zeile nie gab"

EINLEITUNG = """
Eine **partielle Abhängigkeit (PD)** beantwortet die Frage „Wie hängt die Vorhersage vom Vorlauf ab?“, indem sie den Vorlauf für **jede** Zeile auf einen Wert setzt und die Vorhersagen mittelt.
In diesem erfundenen Container-Beispiel gibt es in der Standardeinstellung bei **45 % aller Container** (Import und Leer) nie einen Vorlauf: die Kurve fragt das Modell dort nach Kombinationen, die es nie gab.
**Vorlauf** ist die Zeit in Tagen bis zur Schiffsabfahrt (nur Export und Umschlag haben einen); **Verweilzeit** ist die Zeit in Tagen von der Ankunft eines Containers im Hafen bis zur Abholung, so wie in `verweilzeit-demo`, aus der der Generator kopiert ist.
Gezeigt werden drei Kurven mit drei verschiedenen Wahrheiten (PD, **ALE** (akkumulierte lokale Effekte) und der reale Effekt) und die **ICE-Kurven** (dieselbe Kurve je Zeile) auf den Zeilen, die es nie gab.
Gezählt werden Zeilen, Zufalls-Seeds, Auswertungspunkte und Gruppen; **keine reale Datenquelle**.

**Fünf Kernfragen**, jede mit Zahlen unten: (1) Wie viele Zeilen gibt es nicht? (2) Welche Kurve zeigt den Effekt des Vorlaufs? (3) Was zeigt die ICE auf Zeilen, die es nie gab? (4) Wie viel hängt am Anteil? (5) Fragt die ALE nie nach Unmöglichem?
Wer das Thema zuerst ohne Fachbegriffe verstehen will, öffnet „💬 In einfachen Worten“.
Weiter unten: **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

SO_FUNKTIONIERT = """
1. **Unmögliche Zeile:** eine Kombination, die der Generator nie erzeugt: Import oder Leer mit Vorlauf größer als 0, Export mit Vorlauf außerhalb von 1 bis 10 Tagen, Umschlag außerhalb von 0,5 bis 6, Zollhalt bei etwas anderem als Import. Die Menge der möglichen Kombinationen heißt **Träger** (englisch Support).
2. **PD:** den Vorlauf für alle Zeilen auf einen Wert setzen, Vorhersagen mitteln, für mehrere Werte wiederholen. **ICE:** dieselbe Kurve für jede Zeile einzeln; die PD ist ihr Mittel.
3. **ALE:** den Vorlauf in Intervalle teilen (Quantile), je Intervall nur die Zeilen darin um die Intervallgrenzen verschieben, die Differenzen mitteln und aufsummieren: sie fragt das Modell näher an den Daten, aber nicht ganz (Experiment 5). K ist die angeforderte Zahl der Intervalle; fallen Quantilkanten zusammen, bleiben weniger Intervalle.
4. **Wahrheit je Funktional:** die PD der wahren Funktion μ, die ALE von μ und der **reale Effekt** (Mittel von μ nur über die Zeilen, in denen der Vorlauf vorkommt) sind drei verschiedene Zahlen.
"""

EXP1_CAPTION = "Zähleinheit: Zeilen (Anteile aller Container; exakt in Brüchen, daneben gezählt an einer Stichprobe von 6 000 Zeilen). Gesetzt wird ein Merkmal für alle Zeilen."
EXP2_CAPTION = ("Zähleinheit: Zufalls-Seeds (je Zufalls-Seed ein Trainingssatz und 3 000 frische Auswertungszeilen), Gruppen (6 getrennte Zufalls-Seeds, fünf Gruppen; Spannen über die Gruppen, nie ein Punktwert); eingefroren "
                "(`tools/ukb_make_grid.py`). Anstiege sind Anstiege der ln-Verweilzeit von Vorlauf 2 auf 5 (drei Tage), keine Steigungen je Tag.")
EXP3_CAPTION = ("Zähleinheit: Zufalls-Seeds und Zeilen (je Zufalls-Seed 3 000 frische Auswertungszeilen; Anstieg und flache Kurven an den ersten 2 000 davon, davon die Import- und Leer-Zeilen; der Fehler bei Vorlauf 5 an allen 3 000: "
                "mittlerer Betrag der Abweichung des Modells von μ, wenn der Vorlauf für alle Zeilen auf 5 gesetzt wird, auf Import- und Leer-Zeilen gegen Export- und Umschlag-Zeilen); Wahrheit auf den Import- und Leer-Zeilen: Anstieg exakt 0. "
                "Ein gutes R² auf natürlichen Zeilen sagt nichts über Zeilen, die es nie gab.")
EXP4_CAPTION = "Zähleinheit: Zufalls-Seeds (n = 1 000 Trainingszeilen, 6 000 Auswertungszeilen, fünf Gruppen zu je 4 Zufalls-Seeds). Die Wahrheitskurven stehen geschlossen; PD = (1 − Anteil) · realer Effekt gilt in der geschlossenen Form. Die Tabelle zeigt die eingefrorene Stichprobe der Auswertungszeilen und weicht davon leicht ab (Anteil 0,7: 0,235 gegen 0,233)."
EXP5_CAPTION = ("Zähleinheit: Auswertungspunkte der ALE (Kanten der Intervalle) und Zufalls-Seeds (n = 1 000, fünf Gruppen zu je 4 Zufalls-Seeds). Das erste Intervall enthält alle Zeilen mit Vorlauf 0. "
                "K ist die angeforderte Zahl der Intervalle: bei 45 % Zeilen ohne Vorlauf fallen die unteren Quantilkanten bei 0 zusammen, es bleiben bei K = 5 / 10 / 20 / 50 nur 3 / 6 / 11 bis 12 / 27 bis 30 Intervalle, "
                "und bei K = 20 stammen 88 bis 99 % der unmöglichen Auswertungspunkte aus dem ersten Intervall (3 000 Auswertungszeilen je Zufalls-Seed, 20 Zufalls-Seeds je Messung).")

NEGATIV_2 = ("Ehrlicher Negativbefund: gegen den realen Effekt ist die ALE näher als die gewöhnliche PD (Faktor 1,4 bis 5,4 über die drei trainierten Klassen, Zweitmessung 1,6 bis 5,0), aber nicht durchgehend näher als die PD, nur über die möglichen Zeilen gemittelt "
             "(PD über den Träger; setzt voraus, dass der Träger bekannt ist). Bei n = 3 000 und fünf Gruppen liegt sie beim Netz in 5 von 5 Gruppen näher am realen Effekt als die ALE (Abweichung 0,027 bis 0,041 gegen 0,035 bis 0,055; Zweitmessung 0,029 bis 0,037 gegen 0,039 bis 0,049), "
             "beim Boosting ist dagegen die ALE näher (in 5 von 5 Gruppen der Erstmessung, in 4 von 5 der Zweitmessung), beim Random Forest wechselt es (PD über den Träger näher in 1 von 5 bzw. 3 von 5 Gruppen); "
             "das Verhältnis zweier kleiner Abweichungen streut zwischen Gruppen stark und ist kein fester Faktor.")
NEGATIV_3 = "Ehrlicher Negativbefund: beim Netz sind einige der unmöglichen Kurven flach, das ist keine Entwarnung: sein Fehler bei Vorlauf 5 ist auf unmöglichen Zeilen bei n = 3 000 trotzdem ein Vielfaches des Fehlers auf möglichen."
NEGATIV_4 = ("Ehrlicher Negativbefund: ohne unmögliche Zeilen (Anteil 0) gibt es keinen Vorteil der ALE, der für beide Klassen gilt (die Reihung hängt von der Klasse ab); bei 0,9 sind nur 10 % der Zeilen mit Vorlauf, "
             "die ALE ist beim Boosting in 4 von 5 Gruppen schlechter als die PD, beim Netz in 4 von 5 besser (Deutung des Boosting-Befunds: kaum noch Intervalle mit Daten, nicht getrennt gemessen).")
NEGATIV_5 = ("Ehrlicher Negativbefund: die ALE fragt nicht nie nach Unmöglichem: ein Viertel ihrer Auswertungspunkte (K = 20 angefordert, tatsächlich 11 bis 12 Intervalle) sind unmögliche Kombinationen, "
             "fast alle davon im ersten Intervall, und im ersten Intervall steht ein erfundener Effekt.")

GRENZEN_MARKDOWN = """
| Annahme | Was passiert, wenn sie verletzt ist | Offen |
|---|---|---|
| **Harte strukturelle Abhängigkeit** | Der Vorlauf ist bei Import und Leer exakt 0 (Nullen, ein Träger mit Lücken). Weiche Abhängigkeit (stetige Korrelation) ist hier nicht gemessen. | weiche Abhängigkeit |
| **Regression, ein Merkmal** | Gemessen ist die Wirkung des Vorlaufs auf die Log-Verweilzeit; keine Klassifikation, keine kategorialen Merkmale (ALE für Art, Kunde), keine zweidimensionale ALE oder PD zweier Merkmale. | Klassifikation, kategoriale und zweidimensionale Kurven |
| **Vier Modellklassen mit festen Hyperparametern** | Boosting (150 Bäume), Random Forest (100 Bäume) und das lineare Modell sind nicht abgestimmt, das Netz (16 Einheiten) ist auf R² gegen μ abgestimmt; die Hyperparameter sind nicht variiert, scikit-learn-Modelle sind eingefroren. | andere Hyperparameter |
| **Kurvenabweichung und Anstieg** | Betragsmaß ist die mittlere Abweichung der Kurve über die Punkte 3 bis 6 und der Anstieg von Vorlauf 2 auf 5; ein Rangmaß der Kurven wäre trivial (alle Kurven sind monoton). | Rangmaße der Wichtigkeit: andere Demo |
| **Ursache nicht untersucht** | Warum ein Modell auf unmöglichen Zeilen einen Effekt erfindet (Baumform, Extrapolationsregel), ist nicht untersucht; Deutungen stehen als Deutung da. | Ursache |
| **Zählungen aus Zufalls-Seeds, eine Plattform** | Aussagen als Spanne über Gruppen. Gemessen unter Windows mit Python 3.14; Linux-Lauf (Docker, Python 3.12) am 2026-11-20 grün. | - |
| **Nutzen von PD, ICE und ALE** | In den gelesenen Quellen gab es keine Messung, ob PD, ICE oder ALE in Betrieb, Logistik oder Recht zu besseren Entscheidungen führen; hier wird nichts dazu behauptet. Ob Menschen mit solchen Kurven besser entscheiden, zeigt diese Seite nicht; siehe die Lehrseite zu Menschenstudien. | nicht belegt |
"""

MATHE_BLOECKE = [
    (r"""**Partielle Abhängigkeit (PD).** Den Vorlauf für alle Zeilen auf $z$ setzen und mitteln (Friedman 2001; Definition in Goldstein u. a. 2015 und Apley/Zhu 2020):""",
     r"""\mathrm{PD}(z) \;=\; \frac{1}{n}\sum_{i=1}^{n} f\bigl(z,\, x_{\setminus v}^{(i)}\bigr)"""),
    (r"""**ICE** ist der Summand je Zeile, die PD ihr Mittel (Goldstein u. a. 2015, S. 5). **ALE** summiert bedingte lokale Differenzen (Apley/Zhu 2020) und zentriert; für die wahre Funktion $\mu$ (in der Formel steht $f$; für die wahre Funktion ist $f = \mu$) gilt (Satz 1 dort) bis auf die additive Konstante der Zentrierung die Form mit der bedingten Ableitung:""",
     r"""\mathrm{ALE}(z) \;=\; \int_{z_0}^{z} \mathbb{E}\!\left[\frac{\partial f}{\partial v}\,\Big|\, v = t\right] dt"""),
    (r"""**Der reale Effekt** mittelt nur über die Zeilen, in denen der Vorlauf vorkommt (Export und Umschlag). Weil Import und Leer (Anteil $p_0$) flach sind, gilt exakt""",
     r"""\mathrm{PD}(z) - \mathrm{PD}(z_0) \;=\; (1 - p_0)\,\bigl(\mathrm{real}(z) - \mathrm{real}(z_0)\bigr)"""),
    (r"""**Die PD der wahren Funktion** für den Anstieg von Vorlauf 2 auf 5 (Export Anteil 0,30, Umschlag 0,25 bei $p_0 = 0{,}45$; die Mediane sind $0{,}6 + 0{,}85v$ und $0{,}3 + 0{,}8v$):""",
     r"""0{,}30 \ln\frac{4{,}85}{2{,}3} + 0{,}25 \ln\frac{4{,}3}{1{,}9} \;=\; 0{,}42801"""),
]
MATHE_SCHLUSS = ("**Vollständigkeit:** der Formelblock ist ein vollständiges Modell der Kurven (Definitionen und die geschlossene Wahrheit); die Implementierung (Intervalle aus Quantilen, Zentrierung der ALE) steht in `ukb_curves.py` "
                 "und ist gegen Brüche, das Integral der bedingten Ableitung und eine Handrechnung geprüft. Das erste Intervall der ALE enthält alle Zeilen mit Vorlauf 0 und verschiebt sie auf eine obere Kante größer als 0: dort fragt auch die ALE eine unmögliche Kombination.")

LITERATUR = """
**Literatur** (Stufe W = im Wortlaut gelesen). Apley, Zhu 2020 (ALE): Gleichungen 1, 2 und 6, Satz 1, Bemerkung 1 (S. 6), Beispiele 1 und 2, Abschnitt zu den Rechenkosten, W. Goldstein u. a. 2015 (ICE): S. 4 und 5, W.
Nicht im Wortlaut gelesen: Friedman 2001 (Scan ohne Textebene; die PD-Definition steht in den beiden gelesenen Arbeiten), Hooker, Mentch, Zhou (nur über Kumar u. a. referiert), Molnar (Zusammenfassung).
Vorbilder für nachrechenbare Vergleiche: XAI-Bench, OpenXAI, Hooker u. a.; **neu ist hier keine Aussage, nur die nachrechenbare Vorführung mit Zähleinheiten und Orakeln.**
"""
