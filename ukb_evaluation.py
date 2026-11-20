"""Verdikte als reine Funktionen: jede liefert (Art, Text), die App ruft `getattr(st, Art)(Text)`. `tests/test_verdicts.py` prüft Schwellen und jede Zahl im Text an handgesetzten Werten.

Zahlen im Text haben das Dezimalkomma. Jede Zahl nennt Zähleinheit (Zeilen, Zufalls-Seeds, Auswertungspunkte, Gruppen) und Voraussetzung; Verhältnisse und Anteile stehen als Spanne über Gruppen.
Der Anstieg „von Vorlauf 2 auf 5“ ist ein Anstieg der ln-Verweilzeit über 3 Tage, keine Steigung je Tag."""

from fractions import Fraction


def de(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def spanne(lo: float, hi: float, nd: int = 3) -> str:
    return de(lo, nd) if de(lo, nd) == de(hi, nd) else f"{de(lo, nd)} bis {de(hi, nd)}"


def prozent(x: float, nd: int = 1) -> str:
    return de(100 * x, nd) + " %"


def bruch(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else str(x.numerator)


def _satz_merkmal(label: str) -> str:
    name, wert = (t.strip() for t in label.split("="))
    return {"Vorlauf": f"den Vorlauf für alle Container auf {wert} Tage", "Art": f"die Art für alle Container auf {wert}", "Zoll": "den Zollhalt für alle Container auf 1"}[name]


# ------------------------------------------------------------------------------------------------------- Experiment 1
def verdict_karte(label: str, r: dict) -> tuple[str, str]:
    """Anteil unmöglicher Zeilen. `r` ist das Ergebnis von `ukb_experiments.exp1`."""
    ex = r["exakt"]
    if ex == 0:
        return "success", f"Setzt man {_satz_merkmal(label)}, ist jede Kombination möglich (exakt 0)."
    text = (f"Setzt man {_satz_merkmal(label)}, ist die Kombination bei {de(100 * float(ex), 1)} % der Container eine, die es nie gab (exakt {bruch(ex)} bei einem Anteil von {de(100 * float(r['p0']), 0)} % Containern "
            f"ohne Vorlauf); gezählt an einer Stichprobe von {r['n']} Zeilen: {de(100 * r['stichprobe'], 1)} % (Standardfehler {de(100 * r['se'], 1)} Prozentpunkte).")
    return "warning", text


# ------------------------------------------------------------------------------------------------------- Experiment 2
def verdict_kurven(r: dict, wahr: dict) -> tuple[str, str]:
    """PD, real und ALE: eine Kurve, drei Antworten. `wahr` sind die geschlossenen Wahrheiten (PD, real, ALE) bei 45 %."""
    g = r["n_gruppen"]
    text = (f"{r['klasse']}, n = {r['n']}: Anstieg des Vorlaufs von 2 auf 5 in der PD {spanne(*r['sl_pd'], 2)}, in der ALE {spanne(*r['sl_ale'], 2)} (fünf Gruppen zu je 6 Zufalls-Seeds); "
            f"wahr in der PD {de(wahr['pd'], 2)}, in der ALE {de(wahr['ale'], 2)}, real {de(wahr['real'], 2)} (Wahrheit je Funktional bei 45 % Containern ohne Vorlauf). "
            f"Die PD des Modells liegt in {r['zwischen']} von {g} Gruppen zwischen der PD der Wahrheit und dem realen Effekt. "
            f"Gegen die eigene Wahrheit liegt die ALE {spanne(*r['verh_own'], 1)}-mal näher als die PD, gegen den realen Effekt {spanne(*r['verh_real'], 1)}-mal; die ALE ist gegen die eigene Wahrheit "
            f"in {spanne(100 * r['naeher_own'][0], 100 * r['naeher_own'][1], 0)} % der Zufalls-Seeds näher, gegen den realen Effekt in {spanne(100 * r['naeher_real'][0], 100 * r['naeher_real'][1], 0)} %.")
    if r["klasse"] == "lineares Modell":
        return "warning", text + " Beim linearen Modell sind PD, ALE und die PD über die Stütze gleich: ein additives Modell kann nicht zwischen möglichen und unmöglichen Zeilen unterscheiden."
    return "info", text


# ------------------------------------------------------------------------------------------------------- Experiment 3
def verdict_ice(r: dict) -> tuple[str, str]:
    """ICE auf Zeilen, die es nie gab (Import und Leer; die wahre Funktion ist dort exakt flach)."""
    text = (f"{r['klasse']}, n = {r['n']}: auf Import- und Leer-Zeilen im Mittel ein Anstieg von Vorlauf 2 auf 5 von {spanne(*r['ice_imp'], 2)} statt 0 (Betrag, fünf Gruppen zu je 6 Zufalls-Seeds, an 2 000 Auswertungszeilen je Zufalls-Seed), "
            f"{spanne(*r['flach'], 1)} % der Kurven flach (Anstieg unter 0,1); der Fehler bei Vorlauf 5 ist auf unmöglichen Zeilen {spanne(*r['fehler_verh'], 1)}-mal so groß wie auf möglichen (an 3 000 Auswertungszeilen je Zufalls-Seed); "
            f"R² gegen μ auf natürlichen Zeilen {spanne(*r['r2'], 2)}.")
    return "warning", text


def verdict_linear_live(r: dict) -> tuple[str, str]:
    """Das lineare Modell live: alle ICE-Kurven haben denselben Anstieg."""
    return "info", (f"Lineares Modell, live auf {r['n']} Zeilen gerechnet: Anstieg von Vorlauf 2 auf 5 für alle 3 000 Auswertungszeilen zwischen {de(r['sl_min'], 3)} und {de(r['sl_max'], 3)} "
                    f"(je Tag {de(r['je_tag'], 3)} in ln-Einheiten, also ein fester Prozentzuschlag auf die Verweilzeit); PD {de(r['pd'], 3)}, ALE {de(r['ale'], 3)}, PD über die Stütze {de(r['pd_stuetze'], 3)} (ein Zufalls-Seed; die Gruppenmittel des Experiments 2 liegen bei 0,64 bis 0,65).")


# ------------------------------------------------------------------------------------------------------- Experiment 4
def verdict_dosis(r: dict) -> tuple[str, str]:
    """Dosis: der Anteil ohne Vorlauf. Anker bei den Rasterwerten 0 und 0,9 und die exakte Beziehung bei dem gewählten Anteil."""
    z = {x["p0"]: x for x in r["zeilen"]}
    live = r["live"]
    n0, n9 = z[0.0], z[0.9]
    text = (f"{r['klasse']}: bei Anteil 0 liegt die PD der Wahrheit auf dem realen Effekt ({de(n0['pd_t'], 3)} gegen {de(n0['real'], 3)}); bei 0,9 ist sie {de(n9['pd_t'], 3)} gegen {de(n9['real'], 3)}. "
            f"Exakt gilt PD = (1 − Anteil) · real: beim gewählten Anteil {de(live['p0'], 2)} ist das Verhältnis {de(live['verhaeltnis'], 3)}. "
            f"Abweichung der Modellkurve vom realen Effekt bei Anteil 0 (PD gegen ALE, Spanne über fünf Gruppen zu je 4 Zufalls-Seeds): {spanne(*n0['eP'], 2)} gegen {spanne(*n0['eA'], 2)}; "
            f"bei 0,9: {spanne(*n9['eP'], 2)} gegen {spanne(*n9['eA'], 2)}.")
    return "info", text


# ------------------------------------------------------------------------------------------------------- Experiment 5
def verdict_faecher(r: dict) -> tuple[str, str]:
    """ALE ist nicht gratis: Zahl der Fächer K."""
    text = (f"{r['klasse']}, K = {r['K']} (angefordert; bei 45 % Zeilen ohne Vorlauf bleiben weniger Fächer): {spanne(100 * r['inv'][0], 100 * r['inv'][1], 1)} % der Auswertungspunkte der ALE sind unmögliche Kombinationen; im ersten Fach beträgt der Effekt "
            f"{spanne(*r['eff0_m'], 2)} statt {spanne(*r['eff0_t'], 2)} bei der wahren Funktion; das erste Fach reicht bis Vorlauf {spanne(*r['hi0'], 2)}; "
            f"Abweichung der ALE gegen den realen Effekt {spanne(*r['eA_real'], 2)} (n = 1000, fünf Gruppen zu je 4 Zufalls-Seeds).")
    return ("warning" if min(r["hi0"]) >= 2.0 else "info"), text
