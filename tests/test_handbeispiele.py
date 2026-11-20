"""Handbeispiele von Hand, exakt in Brüchen: Anteile unmöglicher Zeilen (Aufzählung der Zellen gegen die geschlossene Rechnung der Demo), Mini-Welt (PD, ICE, ALE), Generator (Verhältnis PD zu real),
Dosis (PD-Anstieg = (1 − Anteil) · realer Anstieg). Alles hier ist erfunden (kleine Tabellen, Brüche)."""

from fractions import Fraction as Fr

import numpy as np
import pytest

import helpers_exact as H
import ukb_curves as CV
import ukb_feasible as FE
import ukb_generator as G


# ---------------------------------------------------------------------------------------------- Teil 1: Anteile unmöglicher Zeilen
def test_teil1_anteile_von_hand():
    p0 = Fr(9, 20)
    soll = {("vorlauf", 4): Fr(9, 20), ("art", 0): Fr(11, 20), ("art", 1): Fr(26, 55), ("art", 2): Fr(7, 12), ("art", 3): Fr(51, 80), ("zoll", 1): Fr(13, 20)}
    for (merkmal, wert), s in soll.items():
        assert H.anteil_enum(merkmal, wert, p0) == s == FE.unmoeglich_anteil(merkmal, wert, p0), (merkmal, wert)
    # Handrechnung: Art auf Export: 9/20 + 1/4 · 1/11 = 26/55; auf Umschlag: 9/20 + 3/10 · 4/9 = 7/12; auf Leer: 11/20 + 7/20 · 1/4 = 51/80
    assert Fr(9, 20) + Fr(1, 4) * Fr(1, 11) == Fr(26, 55) and Fr(9, 20) + Fr(3, 10) * Fr(4, 9) == Fr(7, 12) and Fr(11, 20) + Fr(7, 20) * Fr(1, 4) == Fr(51, 80)
    assert [float(v) for v in soll.values()] == pytest.approx([0.45, 0.55, 0.4727, 0.5833, 0.6375, 0.65], abs=5e-5)


@pytest.mark.parametrize("p0", [Fr(0), Fr(1, 5), Fr(9, 20), Fr(7, 10), Fr(9, 10)])
def test_teil1_geschlossen_gleich_aufzaehlung_fuer_viele_werte_und_anteile(p0):
    for v in (0, Fr(3, 4), Fr(5, 4), Fr(9, 4), Fr(17, 4), Fr(23, 4), Fr(25, 4), Fr(39, 4), Fr(41, 4), 12):      # Mitten der Halbtag-Zellen (die Ränder prüft der nächste Test)
        assert H.anteil_enum("vorlauf", v, p0) == FE.unmoeglich_anteil("vorlauf", v, p0), (p0, v)
    for a in range(4):
        assert H.anteil_enum("art", a, p0) == FE.unmoeglich_anteil("art", a, p0), (p0, a)
    for z in (0, 1):
        assert H.anteil_enum("zoll", z, p0) == FE.unmoeglich_anteil("zoll", z, p0)
    assert sum(u for u, _ in FE.unmoeglich_je_art("art", 2, p0)) == FE.unmoeglich_anteil("art", 2, p0)


def test_teil1_raender_der_intervalle_gehoeren_zur_stuetze():
    """Export 1 bis 10 und Umschlag 0,5 bis 6 sind abgeschlossen: der Rand gilt als möglich (die Demo und die geschlossene Rechnung rechnen so)."""
    for v, art, soll in ((1.0, 1, True), (10.0, 1, True), (0.99, 1, False), (10.01, 1, False), (0.5, 2, True), (6.0, 2, True), (0.49, 2, False), (6.01, 2, False)):
        assert bool(FE.gueltig(np.array([[art, 0, 0, 0, v, 0.0]]))[0]) == soll, (art, v)
    p0 = Fr(9, 20)
    assert FE.unmoeglich_anteil("vorlauf", 6, p0) == Fr(9, 20) and FE.unmoeglich_anteil("vorlauf", Fr(1, 2), p0) == Fr(9, 20) + Fr(6, 11) * Fr(11, 20)


def test_teil1_gueltigkeitspruefung_gegen_aufzaehlung_und_stichprobe():
    """gueltig() der Demo gegen die Intervallregel der Zellen (je ein Punkt in der Mitte jeder Zelle) und der exakte Anteil gegen eine Stichprobe (Band, 4,5 Standardfehler)."""
    for art, zoll, zelle, w in H.zellen(Fr(9, 20)):
        v = 0.0 if zelle is None else float(zelle[0] + zelle[1] / 2)
        assert bool(FE.gueltig(np.array([[art, 0, zoll, 0, v, 0.0]]))[0]) == H.gueltig_zelle(art, zoll, zelle)
    from ukb_rng import SplitMix64
    X, _ = G.draw(8000, SplitMix64(123))
    for merkmal, wert, spalte in (("vorlauf", 4, 4), ("art", 1, 0), ("zoll", 1, 2)):
        Z = X.copy()
        Z[:, spalte] = wert
        ex = float(FE.unmoeglich_anteil(merkmal, wert, Fr(9, 20)))
        assert 1 - FE.gueltig(Z).mean() == pytest.approx(ex, abs=4.5 * np.sqrt(ex * (1 - ex) / 8000))


# ---------------------------------------------------------------------------------------------- Teil 2: Mini-Welt (drei Zellen, Brüche)
ZEILEN = [(0, Fr(0), Fr(3, 5)), (1, Fr(1), Fr(2, 15)), (1, Fr(2), Fr(2, 15)), (1, Fr(3), Fr(2, 15))]          # (Import, 0), (Export, 1), (Export, 2), (Export, 3)


def wahr(a, v):
    return Fr(3) if a == 0 else 1 + 2 * v                                                                     # Import konstant 3, Export 1 + 2v


def modell(a, v):
    return 3 + v if a == 0 else 1 + 2 * v                                                                     # stimmt auf der Stütze, erfindet für (Import, v > 0) die Steigung 1


def test_teil2_mini_welt_pd_ice_ale():
    assert sum(w for *_, w in ZEILEN) == 1
    pd_w, pd_m = H.pd_exakt(wahr, ZEILEN, [1, 3]), H.pd_exakt(modell, ZEILEN, [1, 3])
    assert pd_w[1] - pd_w[0] == Fr(8, 5) == (1 - Fr(3, 5)) * 4                                                # PD 1 → 3: 3/5 · 0 + 2/5 · 4 = 8/5 = (1 − Anteil) · real
    assert pd_m[1] - pd_m[0] == Fr(14, 5) == Fr(3, 5) * 2 + Fr(8, 5)                                         # Modell: 3/5 · 2 + 8/5 = 14/5
    export = [z for z in ZEILEN if z[0] == 1]
    ex_gew = sum(w for *_, w in export)
    assert sum(w * (wahr(a, 3) - wahr(a, 1)) for a, v, w in export) / ex_gew == 4 == sum(w * (modell(a, 3) - modell(a, 1)) for a, v, w in export) / ex_gew   # über die Export-Zeilen: 4 für beide
    # ICE: Import-Zeile wahr 0, Modell 2; Export-Zeile 4 für beide; Mittel der ICE-Steigungen = PD-Steigung
    assert H.ice_exakt(wahr, ZEILEN[0], 1, 3) == 0 and H.ice_exakt(modell, ZEILEN[0], 1, 3) == 2 and H.ice_exakt(wahr, ZEILEN[1], 1, 3) == 4 == H.ice_exakt(modell, ZEILEN[1], 1, 3)
    assert sum(w * H.ice_exakt(modell, z, 1, 3) for z in ZEILEN for w in [z[2]]) == pd_m[1] - pd_m[0]
    # ALE (Fächer (0,1], (1,2], (2,3], das erste schließt 0 ein): Effekte wahr 4/11, 2, 2; Modell 13/11, 2, 2; erstes Fach Modell (3/5 · 1 + 2/15 · 2) / (11/15)
    kanten = [Fr(0), Fr(1), Fr(2), Fr(3)]
    assert H.ale_exakt(wahr, ZEILEN, kanten) == [Fr(4, 11), 2, 2] and H.ale_exakt(modell, ZEILEN, kanten) == [Fr(13, 11), 2, 2]
    assert (Fr(3, 5) * 1 + Fr(2, 15) * 2) / Fr(11, 15) == Fr(13, 11)
    assert sum(H.ale_exakt(wahr, ZEILEN, kanten)[1:]) == 4 == sum(H.ale_exakt(modell, ZEILEN, kanten)[1:])      # Anstieg 1 → 3: 4 für beide, gleich dem realen; im ersten Fach weicht auch die ALE ab


def test_teil2_float_funktionen_der_demo_gleich_den_bruechen():
    """`pd_kurve`, `ice_anstiege` und `ale_1d` der Demo auf der Mini-Welt (Zeilen mit Gewichten als Wiederholung: 9 Import, je 2 Export, Gesamtgewicht 15) gegen die Brüche."""
    X = np.array([[0, 0, 0, 0, 0.0, 0]] * 9 + [[1, 0, 0, 0, 1.0, 0]] * 2 + [[1, 0, 0, 0, 2.0, 0]] * 2 + [[1, 0, 0, 0, 3.0, 0]] * 2)
    f = lambda Z: np.where(Z[:, 0] == 0, 3.0 + Z[:, 4], 1 + 2 * Z[:, 4])
    pd = CV.pd_kurve(f, X, 4, np.array([1.0, 3.0]))
    assert pd[1] - pd[0] == pytest.approx(14 / 5)
    assert CV.ice_anstiege(f, X, 4, 1.0, 3.0) == pytest.approx([2.0] * 9 + [4.0] * 6)
    e, a, inv, eff = CV.ale_1d(f, X, 4, edges=[0.0, 1.0, 2.0, 3.0])
    assert eff == pytest.approx([13 / 11, 2, 2]) and a[3] - a[1] == pytest.approx(4)


# ---------------------------------------------------------------------------------------------- Teil 3: Generator
def test_teil3_pd_real_und_ale_der_wahren_funktion_geschlossen():
    ln = np.log
    pd = 0.30 * ln(4.85 / 2.3) + 0.25 * ln(4.3 / 1.9)
    pd_c, real_c, ale_c = CV.wahrheiten_anstieg_2_5()
    assert pd_c == pytest.approx(pd, abs=1e-12) and pd_c == pytest.approx(0.42801, abs=5e-6)                  # 0,30·ln(4,85/2,3) + 0,25·ln(4,3/1,9)
    assert real_c == pytest.approx(0.77820, abs=5e-6) and pd_c / real_c == pytest.approx(0.55, abs=1e-12)    # real = PD / 0,55: das Verhältnis ist der Anteil möglicher Zeilen
    assert ale_c == pytest.approx(0.78685, abs=5e-6) and ale_c - real_c == pytest.approx(0.0087, abs=5e-5)    # ALE minus real = 0,0087
    assert CV.pd_wahrheit()[3] == pd_c and CV.pd_wahrheit().shape == (5,)


def test_teil3_wahrheit_geschlossen_gegen_stichprobe_im_band():
    from ukb_rng import SplitMix64
    X, _ = G.draw(40000, SplitMix64(7))
    pd = CV.pd_kurve(G.wahrheit, X, 4, np.array([2.0, 5.0]))
    real = CV.pd_kurve(G.wahrheit, X[np.isin(X[:, 0], (1, 2))], 4, np.array([2.0, 5.0]))
    assert pd[1] - pd[0] == pytest.approx(0.42801, abs=0.01) and real[1] - real[0] == pytest.approx(0.77820, abs=0.01)


# ---------------------------------------------------------------------------------------------- Teil 4: Dosis
@pytest.mark.parametrize("p0", [Fr(0), Fr(1, 5), Fr(9, 20), Fr(7, 10), Fr(9, 10)])
def test_teil4_pd_anstieg_ist_eins_minus_anteil_mal_real(p0):
    """Unabhängig gerechnet: PD aus den Anteilen Export und Umschlag (Brüche der Zellen), real aus Export und Umschlag allein; Verhältnis exakt 1 − p0."""
    pe, pu = Fr(6, 11) * (1 - p0), Fr(5, 11) * (1 - p0)
    e = np.log(0.6 + 0.85 * 5) - np.log(0.6 + 0.85 * 2)
    u = np.log(0.3 + 0.8 * 5) - np.log(0.3 + 0.8 * 2)
    pd = float(pe) * e + float(pu) * u
    real = (float(pe) * e + float(pu) * u) / float(pe + pu)
    assert float(CV.pd_wahrheit(p0=float(p0))[3]) == pytest.approx(pd, abs=1e-12)
    if p0 < 1:
        assert pd / real == pytest.approx(float(1 - p0), abs=1e-12) and float(CV.real_wahrheit(p0=float(p0))[3]) == pytest.approx(real, abs=1e-12)
    assert real == pytest.approx(0.77820, abs=5e-6)                                                       # der reale Effekt hängt nicht am Anteil
    assert float(CV.ale_wahrheit(p0=float(p0))[3]) == pytest.approx(0.78685, abs=5e-6)                  # und die ALE auch nicht (der Anteil kürzt sich heraus)
