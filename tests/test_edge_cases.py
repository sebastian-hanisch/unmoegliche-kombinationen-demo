"""Randfälle: Anteil 0 und 0,9, n = 2, ein Fach (K = 1), konstante Spalte, Vorlauf ohne Variation, Werte außerhalb der Daten, Gültigkeitsränder, Anteil nicht auf dem Raster."""

from fractions import Fraction as Fr

import numpy as np
import pytest

import ukb_curves as CV
import ukb_evaluation as ev
import ukb_experiments as E
import ukb_feasible as FE
import ukb_generator as G
import ukb_models as M
from ukb_rng import SplitMix64


def test_anteil_0_hat_keine_unmoeglichen_zeilen_beim_vorlauf_aus_dem_traeger():
    assert FE.unmoeglich_anteil("vorlauf", 4, Fr(0)) == 0 and E.exp1("Vorlauf = 4", 0.0)["stichprobe"] == 0.0
    assert FE.unmoeglich_anteil("vorlauf", 4, Fr(9, 10)) == Fr(9, 10)


def test_vorlauf_null_ist_fuer_export_und_umschlag_unmoeglich():
    assert FE.unmoeglich_anteil("vorlauf", 0, Fr(9, 20)) == Fr(11, 20)                                  # nur Import und Leer (45 %) sind möglich
    assert FE.unmoeglich_anteil("vorlauf", 12, Fr(9, 20)) == 1


def test_n_gleich_2_und_konstante_spalte():
    X = np.array([[1, 0, 0, 0, 2.0, 0.0], [2, 1, 0, 0, 3.0, 0.0]])
    f = lambda Z: Z[:, 4] * 2.0
    assert CV.pd_kurve(f, X, 4) == pytest.approx(2 * CV.GITTER) and CV.ice_anstiege(f, X, 4) == pytest.approx([6.0, 6.0])
    e, a, inv, eff = CV.ale_1d(f, X, 4, 20)                                                             # nur zwei verschiedene Werte: die Quantile werden interpoliert, 20 Fächer, kein Fehler
    assert len(e) - 1 == 20 and np.isfinite(a).all()
    Xk = np.zeros((10, 6))
    Xk[:, 4] = 3.0                                                                                      # konstante Spalte: keine Fächer, Meldung statt Zahl
    with pytest.raises(ValueError, match="ALE nicht definiert"):
        CV.ale_1d(lambda Z: Z[:, 4], Xk, 4, 20)


def test_ein_fach_k_gleich_1():
    X, _ = G.draw(300, SplitMix64(2))
    e, a, inv, eff = CV.ale_1d(lambda Z: 2 * Z[:, 4], X, 4, 1)
    assert len(eff) == 1 and eff[0] == pytest.approx(2 * (X[:, 4].max() - X[:, 4].min()))


def test_vorlauf_ohne_variation_alle_zeilen_import():
    X = np.zeros((20, 6))                                                                                # alle Import, Vorlauf 0: die ALE ist nicht definiert
    with pytest.raises(ValueError, match="ALE nicht definiert"):
        CV.ale_1d(lambda Z: Z[:, 4], X, 4, 20)
    assert CV.pd_kurve(lambda Z: Z[:, 4], X, 4) == pytest.approx(CV.GITTER)                              # die PD wird trotzdem gerechnet: sie fragt nur unmögliche Kombinationen


def test_gitter_ausserhalb_der_daten_pd_wird_extrapoliert_nicht_abgeschnitten():
    X, y = G.draw(500, SplitMix64(8))
    f = M.fit_linear(X, y)
    pd = CV.pd_kurve(f, X, 4, np.array([20.0, 40.0]))
    assert pd[1] - pd[0] == pytest.approx(20 * (CV.anstieg(CV.pd_kurve(f, X, 4))[0] / 1.0), rel=1e-9)   # linear: Steigung je Tag mal 20 Tage


def test_p0_auf_dem_slider_raster_ist_exakt_ein_bruch():
    for k in range(19):
        assert E.p0_bruch(k * 0.05) == Fr(k, 20)


def test_unbekanntes_merkmal_wird_abgelehnt():
    with pytest.raises(ValueError):
        FE.unmoeglich_anteil("gewicht", 1, Fr(9, 20))
    assert ev.bruch(E.exp1("Zoll = 1", 0.45)["exakt"]) == "13/20"


def test_wahrheit_bei_anteil_0_9_bleibt_endlich_und_positiv():
    w = E.wahrheiten(0.9)
    assert 0 < w["pd"] < w["real"] and w["verhaeltnis"] == pytest.approx(0.1)
