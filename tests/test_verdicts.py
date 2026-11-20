"""Verdikte an von Hand gesetzten Werten: jede Schwelle der Art und jede Zahl im Text steht hier. Die Verdikte sind reine Funktionen (Art, Text)."""

from fractions import Fraction as Fr

import ukb_evaluation as ev


def test_zahlformate():
    assert ev.de(0.0996, 3) == "0,100" and ev.de(-0.91, 2) == "−0,91" and ev.spanne(0.68, 0.75, 2) == "0,68 bis 0,75" and ev.spanne(0.1, 0.1004) == "0,100"
    assert ev.prozent(0.375) == "37,5 %" and ev.bruch(Fr(26, 55)) == "26/55" and ev.bruch(Fr(0)) == "0"


def _r1(ex, p0=Fr(9, 20), st=0.448, se=0.006):
    return {"exakt": ex, "p0": p0, "stichprobe": st, "se": se, "n": 6000}


def test_karte_nennt_anteil_bruch_und_stichprobe():
    kind, text = ev.verdict_karte("Vorlauf = 4", _r1(Fr(9, 20)))
    assert kind == "warning"
    assert "Setzt man den Vorlauf für alle Container auf 4 Tage" in text and "bei 45,0 % der Container eine, die es nie gab (exakt 9/20 bei einem Anteil von 45 % Containern ohne Vorlauf)" in text
    assert "Stichprobe von 6000 Zeilen: 44,8 % (Standardfehler 0,6 Prozentpunkte)" in text
    assert "die Art für alle Container auf Export" in ev.verdict_karte("Art = Export", _r1(Fr(26, 55)))[1] and "den Zollhalt für alle Container auf 1" in ev.verdict_karte("Zoll = 1", _r1(Fr(13, 20)))[1]


def test_karte_ohne_unmoegliche_zeilen_ist_ein_erfolg():
    kind, text = ev.verdict_karte("Vorlauf = 4", _r1(Fr(0), p0=Fr(0)))
    assert kind == "success" and "jede Kombination möglich (exakt 0)" in text


def _r2(klasse="Boosting"):
    return {"klasse": klasse, "n": 3000, "n_gruppen": 5, "sl_pd": (0.68, 0.746), "sl_ale": (0.757, 0.8), "zwischen": 5, "verh_own": (3.87, 9.34), "verh_real": (1.35, 3.71), "naeher_own": (1.0, 1.0), "naeher_real": (0.67, 1.0)}


WAHR = {"pd": 0.428, "ale": 0.787, "real": 0.778}


def test_kurven_verdikt_nennt_spannen_wahrheit_und_zaehlung():
    kind, text = ev.verdict_kurven(_r2(), WAHR)
    assert kind == "info"
    assert "Boosting, n = 3000: Anstieg des Vorlaufs von 2 auf 5 in der PD 0,68 bis 0,75, in der ALE 0,76 bis 0,80 (fünf Gruppen zu je 6 Zufalls-Seeds)" in text
    assert "wahr in der PD 0,43, in der ALE 0,79, real 0,78" in text and "in 5 von 5 Gruppen zwischen der PD der Wahrheit und dem realen Effekt" in text
    assert "ALE 3,9 bis 9,3-mal näher als die PD, gegen den realen Effekt 1,4 bis 3,7-mal" in text and "in 100 % der Zufalls-Seeds näher, gegen den realen Effekt in 67 bis 100 %" in text


def test_kurven_verdikt_lineares_modell_ist_eine_warnung():
    kind, text = ev.verdict_kurven(_r2("lineares Modell"), WAHR)
    assert kind == "warning" and "PD, ALE und die PD über die Stütze gleich" in text


def test_ice_verdikt():
    r = {"klasse": "Netz", "n": 3000, "ice_imp": (0.21, 0.393), "flach": (6.2, 30.3), "fehler_verh": (9.4, 15.3), "r2": (0.961, 0.965)}
    kind, text = ev.verdict_ice(r)
    assert kind == "warning" and "von 0,21 bis 0,39 statt 0" in text and "6,2 bis 30,3 % der Kurven flach" in text and "9,4 bis 15,3-mal so groß" in text and "R² gegen μ auf natürlichen Zeilen 0,96." in text


def test_linear_live_verdikt():
    kind, text = ev.verdict_linear_live({"n": 3000, "sl_min": 0.665, "sl_max": 0.665, "je_tag": 0.2217, "pd": 0.665, "ale": 0.665, "pd_stuetze": 0.665})
    assert kind == "info" and "live auf 3000 Zeilen" in text and "zwischen 0,665 und 0,665 (je Tag 0,222 in ln-Einheiten, also ein fester Prozentzuschlag auf die Verweilzeit)" in text


def test_dosis_verdikt_nennt_anker_und_exakte_beziehung():
    zeilen = [{"p0": 0.0, "pd_t": 0.778, "real": 0.778, "eP": (0.05, 0.13), "eA": (0.03, 0.07)}, {"p0": 0.9, "pd_t": 0.078, "real": 0.778, "eP": (0.15, 0.21), "eA": (0.11, 0.48)}]
    kind, text = ev.verdict_dosis({"klasse": "Boosting", "zeilen": zeilen, "live": {"p0": 0.45, "verhaeltnis": 0.55}})
    assert kind == "info"
    assert "bei Anteil 0 liegt die PD der Wahrheit auf dem realen Effekt (0,778 gegen 0,778); bei 0,9 ist sie 0,078 gegen 0,778" in text
    assert "beim gewählten Anteil 0,45 ist das Verhältnis 0,550" in text and "0,05 bis 0,13 gegen 0,03 bis 0,07" in text and "bei 0,9: 0,15 bis 0,21 gegen 0,11 bis 0,48" in text


def test_faecher_verdikt_schwelle_erstes_fach():
    r = {"klasse": "Boosting", "K": 5, "inv": (0.318, 0.32), "eff0_m": (-0.28, -0.18), "eff0_t": (0.445, 0.493), "hi0": (2.52, 2.67), "eA_real": (0.297, 0.357), "n_gruppen": 5}
    kind, text = ev.verdict_faecher(r)
    assert kind == "warning" and "31,8 bis 32,0 % der Auswertungspunkte" in text and "Effekt −0,28 bis −0,18 statt 0,45 bis 0,49" in text and "bis Vorlauf 2,52 bis 2,67" in text
    r.update(K=10, hi0=(1.29, 1.39))
    assert ev.verdict_faecher(r)[0] == "info"                                                           # die obere Kante des ersten Fachs liegt unter 2
