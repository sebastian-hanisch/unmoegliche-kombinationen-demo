"""Orakel für PD, ICE und ALE (ORACLE-PLAYBOOK §5): anderer Rechenweg als die Demo.

* ALE der wahren Funktion: geschlossene Gauß-Legendre-Rechnung gegen das Integral der bedingten Ableitung mit scipy (Apley/Zhu 2020 Satz 1, Stufe W) und gegen die gebinnte ALE auf einer deterministischen Population.
* Eigenschaften: PD = Mittel der ICE-Kurven (exakt), Verschiebung der Zielgröße, additive Funktion: PD und ALE gleich bis auf eine Konstante (Apley/Zhu 2020 Bemerkung 1, Stufe W),
  lineares Modell: PD = ALE = PD über die Stütze und alle ICE-Anstiege gleich, Umnummerieren der Zeilen ändert nichts.
Kein Paket `shap`/`lime`; PD, ICE und ALE sind eigene Implementierungen."""

import numpy as np
import pytest

import ukb_curves as CV
import ukb_generator as G
import ukb_models as M
from ukb_rng import SplitMix64


def test_ale_der_wahrheit_gegen_integral_mit_scipy():
    quad = pytest.importorskip("scipy.integrate").quad

    def ableitung(t):
        de = 0.30 / 9 if 1 <= t <= 10 else 0.0
        du = 0.25 / 5.5 if 0.5 <= t <= 6 else 0.0
        return (de * 0.85 / (0.6 + 0.85 * t) + du * 0.8 / (0.3 + 0.8 * t)) / (de + du)
    ale = CV.ale_wahrheit()
    assert ale[3] == pytest.approx(quad(ableitung, 2, 5)[0], abs=1e-9)                                     # 0,78685
    assert ale[1] == pytest.approx(quad(ableitung, 2, 3)[0], abs=1e-9) and ale[4] == pytest.approx(quad(ableitung, 2, 6)[0], abs=1e-9)


def _population():
    """Deterministische Population (keine Zufallsziehung): Anteile genau nach der Formel, Vorlauf auf einem feinen Raster."""
    zeilen = []
    for art, anteil, lo, hi in ((0, 7, None, None), (3, 2, None, None), (1, 6, 1.0, 10.0), (2, 5, 0.5, 6.0)):
        m = anteil * 100
        if lo is None:
            zeilen += [[art, 0, 0, 0, 0.0, 0.0]] * m
        else:
            zeilen += [[art, 0, 0, 0, lo + (hi - lo) * (i + 0.5) / m, 0.0] for i in range(m)]
    return np.array(zeilen, float)


def test_gebinnte_ale_gegen_geschlossene_form_auf_deterministischer_population():
    X = _population()
    # ohne Kundeneffekt: gleiche Ableitung nach dem Vorlauf; Import und Leer sind flach
    mu0 = lambda Z: np.log(np.select([Z[:, 0] == 1, Z[:, 0] == 2], [0.6 + 0.85 * Z[:, 4], 0.3 + 0.8 * Z[:, 4]], default=3.5))
    for K in (20, 100):
        e, a, inv, _ = CV.ale_1d(mu0, X, 4, K)
        assert CV.anstieg(np.interp(CV.GITTER, e, a))[2] == pytest.approx(0.78685, abs=0.02), K              # Toleranz 0,02 (Plan), K = 20 und 100


def test_pd_ist_das_mittel_der_ice_kurven_und_verschiebung_der_zielgroesse():
    X, y = G.draw(500, SplitMix64(3))
    f = M.fit_linear(X, y)
    g = lambda Z: np.sin(Z[:, 4]) * Z[:, 1] + 0.1 * Z[:, 0] * Z[:, 4]
    for fn in (f, g):
        ice = np.array([fn(np.where(np.arange(6) == 4, v, X)) for v in CV.GITTER])
        assert CV.pd_kurve(fn, X, 4) == pytest.approx(ice.mean(axis=1), abs=1e-12)                       # PD = Mittel der ICE-Kurven, exakt
        h = lambda Z, fn=fn: fn(Z) + 2.5
        assert CV.pd_kurve(h, X, 4) == pytest.approx(CV.pd_kurve(fn, X, 4) + 2.5, abs=1e-12)           # Verschiebung der Zielgröße verschiebt die PD
        assert CV.anstieg(CV.ale_an_gitter(h, X)[0]) == pytest.approx(CV.anstieg(CV.ale_an_gitter(fn, X)[0]), abs=1e-12)   # die zentrierte ALE bleibt
    perm = np.random.default_rng(0).permutation(len(X))
    assert CV.pd_kurve(g, X[perm], 4) == pytest.approx(CV.pd_kurve(g, X, 4), abs=1e-12)              # Umnummerieren der Zeilen ändert nichts
    assert CV.ale_1d(g, X[perm], 4, 10)[1] == pytest.approx(CV.ale_1d(g, X, 4, 10)[1], abs=1e-12)


def test_additive_funktion_pd_und_ale_gleich_bis_auf_eine_konstante():
    """Apley/Zhu 2020 Bemerkung 1 (S. 6): für ein additives f sind PD und ALE gleich dem wahren Effekt bis auf eine Konstante."""
    X, _ = G.draw(3000, SplitMix64(9))
    f = lambda Z: np.sqrt(Z[:, 4] + 1.0) + 0.3 * Z[:, 1] - Z[:, 3]                                       # additiv im Vorlauf
    pd = CV.anstieg(CV.pd_kurve(f, X, 4))
    ale = CV.anstieg(CV.ale_an_gitter(f, X, 4, 100)[0])
    wahr = CV.anstieg(np.sqrt(CV.GITTER + 1.0))
    assert pd == pytest.approx(wahr, abs=1e-12) and ale == pytest.approx(wahr, abs=0.02)


def test_lineares_modell_pd_ale_und_stuetze_gleich_und_alle_ice_gleich():
    X, y = G.draw(3000, SplitMix64(4))
    f = M.fit_linear(X, y)
    sl = CV.ice_anstiege(f, X, 4)
    real = np.isin(X[:, 0], (1, 2))
    assert np.ptp(sl) < 1e-9                                                                            # alle ICE-Kurven haben dieselbe Form
    assert CV.anstieg(CV.pd_kurve(f, X, 4))[2] == pytest.approx(sl[0], abs=1e-9)
    assert CV.anstieg(CV.ale_an_gitter(f, X)[0])[2] == pytest.approx(sl[0], abs=1e-9)
    assert CV.anstieg(CV.pd_kurve(f, X[real], 4))[2] == pytest.approx(sl[0], abs=1e-9)
    # kleinste Quadrate an einem einfachen Fall: y = 2 v + 1 ohne Rauschen wird exakt getroffen
    Z = np.zeros((50, 6))
    Z[:, 4] = np.linspace(0, 5, 50)
    Z[:, 1] = 1.0
    g = M.fit_linear(Z, 2 * Z[:, 4] + 1)
    assert g(Z) == pytest.approx(2 * Z[:, 4] + 1, abs=1e-9)
