"""Kopie-Test: `ukb_generator.py` kopiert die Formeln und Anteile aus `verweilzeit-demo` (`vwz_generator.py`, `vwz_constants.py`). Die Werte stammen aus der Quelle (eingefroren am 2026-11-19);
sie fallen auf, sobald die Kopie abweicht. Gezogen wird hier mit SplitMix64, deshalb prüft ein zweiter Teil die Verteilung (Anteile, Träger) im Band."""

import numpy as np
import pytest

import ukb_feasible as FE
import ukb_generator as G
from ukb_rng import SplitMix64


def test_kundeneffekte_und_mu_wie_in_der_quelle():
    assert G.UK[:4].tolist() == pytest.approx([-0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927], abs=1e-12)
    mu, s = G.log_median_and_s(np.array([0, 1, 2, 3]), np.array([0, 5, 7, 11]), np.array([1, 0, 0, 0]), np.array([1, 0, 1, 0]), np.array([0.0, 4.0, 2.5, 0.0]))
    assert mu.tolist() == pytest.approx([1.8256154575589787, 1.3196147424321858, 0.8913094751324775, 2.2364592207808904], abs=1e-12) and s.tolist() == [0.55, 0.25, 0.35, 0.9]


def test_anteile_bei_045_sind_die_der_quelle_und_summieren_zu_eins():
    pa = G.art_wahrscheinlichkeiten(0.45)
    assert pa == pytest.approx(G.ART_P, abs=1e-12) and sum(pa) == pytest.approx(1.0)
    for p0 in (0.0, 0.2, 0.7, 0.9):
        pa = G.art_wahrscheinlichkeiten(p0)
        assert sum(pa) == pytest.approx(1.0) and pa[0] + pa[3] == pytest.approx(p0)


def test_ziehung_deterministisch_und_im_traeger_mit_anteilen_im_band():
    X, y = G.draw(6000, SplitMix64(11))
    X2, y2 = G.draw(6000, SplitMix64(11))
    assert (X == X2).all() and (y == y2).all()
    assert FE.gueltig(X).all()                                                                         # der Generator erzeugt nur mögliche Zeilen
    art = X[:, 0].astype(int)
    for a, p in enumerate(G.ART_P):
        assert (art == a).mean() == pytest.approx(p, abs=4.5 * np.sqrt(p * (1 - p) / 6000))
    assert (X[art == 0, 2] == 1).mean() == pytest.approx(0.25, abs=4.5 * np.sqrt(0.25 * 0.75 / (art == 0).sum()))
    assert ((X[art == 1, 4] >= 1) & (X[art == 1, 4] <= 10)).all() and ((X[art == 2, 4] >= 0.5) & (X[art == 2, 4] <= 6)).all()
    assert X[art == 1, 4].mean() == pytest.approx(5.5, abs=0.15) and X[art == 2, 4].mean() == pytest.approx(3.25, abs=0.15)
    mu = G.wahrheit(X)
    assert np.std(y - mu) > 0.2 and abs(np.mean(y - mu)) < 0.05


def test_anteil_ohne_vorlauf_als_regler():
    for p0 in (0.0, 0.9):
        X, _ = G.draw(4000, SplitMix64(5), p0)
        assert np.isin(X[:, 0], (0, 3)).mean() == pytest.approx(p0, abs=4.5 * np.sqrt(max(p0 * (1 - p0), 1e-12) / 4000) + 1e-12)
