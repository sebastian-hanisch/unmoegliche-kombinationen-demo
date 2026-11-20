"""Claims: jede Zahl der README (Abschnitte „Befunde und Korrekturen“ und „Befunde“) steht hier, mit Bedingung (Tabelle „Zahl → Test“ der README).

Band: eingefrorene Messungen (Gruppenmittel über Zufalls-Seeds) werden auf die in der README genannte Stellenzahl verglichen (Toleranz 0,006 bei zwei, 0,0006 bei drei Stellen); Anteile
von Zufalls-Seeds als Zählung je Gruppe. Die Wahrheitskurven und die Brüche sind exakt (siehe auch `test_handbeispiele.py`)."""

import numpy as np
import pytest

import ukb_data as DATA
import ukb_experiments as E

T2, T3 = 0.006, 0.0006
KL = ("Boosting", "Random Forest", "Netz", "lineares Modell")


def _sp(werte, lo, hi, tol=T2):
    assert min(werte) == pytest.approx(lo, abs=tol) and max(werte) == pytest.approx(hi, abs=tol), (min(werte), max(werte), lo, hi)


def test_k1_anteil_ohne_vorlauf_exakt_045():
    from fractions import Fraction
    import ukb_generator as G
    assert sum(G.art_wahrscheinlichkeiten(Fraction(9, 20))[i] for i in (0, 3)) == Fraction(9, 20) and 0.35 + 0.10 == pytest.approx(0.45)
    r = E.exp1("Vorlauf = 4", 0.45)
    assert str(r["exakt"]) == "9/20" and r["stichprobe"] == pytest.approx(0.45, abs=4.5 * r["se"])


def test_k2_wahrheiten_geschlossen():
    w = E.wahrheiten()
    assert (w["pd"], w["real"], w["ale"]) == pytest.approx((0.42801, 0.77820, 0.78685), abs=5e-6) and w["verhaeltnis"] == pytest.approx(0.55, abs=1e-12)
    assert w["ale"] - w["real"] == pytest.approx(0.0087, abs=5e-5)


def test_k3_anstiege_pd_und_ale_je_klasse_und_gruppenmittel():
    erw = {"Boosting": ((0.68, 0.75), (0.76, 0.80)), "Random Forest": ((0.60, 0.63), (0.75, 0.79)), "Netz": ((0.53, 0.62), (0.81, 0.84)), "lineares Modell": ((0.64, 0.65), (0.64, 0.65))}
    for k, (pd, ale) in erw.items():
        r = E.exp2(k, 3000)
        _sp(r["sl_pd"], *pd)
        _sp(r["sl_ale"], *ale)
        assert r["n_gruppen"] == 5
    zwischen = sum(E.exp2(k, n)["zwischen"] for k in KL for n in (300, 3000))
    assert zwischen == 40                                                                              # alle 40 Gruppenmittel (4 Klassen, 2 Größen, 5 Gruppen) zwischen PD der Wahrheit und real
    r = E.exp2("Boosting", 3000)
    assert r["wahr_pd"] == pytest.approx(0.43, abs=0.006) and r["wahr_real"] == pytest.approx(0.78, abs=0.006)


def test_k4_verhaeltnisse_ale_gegen_pd():
    own = {"Boosting": (3.9, 9.3), "Random Forest": (3.4, 6.2), "Netz": (2.5, 4.1), "lineares Modell": (1.3, 1.4)}
    real = {"Boosting": (1.4, 3.7), "Random Forest": (2.5, 4.1), "Netz": (2.8, 5.4), "lineares Modell": (1.0, 1.0)}
    for k in KL:
        r = E.exp2(k, 3000)
        _sp(r["verh_own"], *own[k], tol=0.06)
        _sp(r["verh_real"], *real[k], tol=0.06)
    own300 = {"Boosting": (1.3, 3.7), "Random Forest": (0.44, 1.5), "Netz": (0.91, 1.2), "lineares Modell": (1.0, 1.5)}
    for k, (lo, hi) in own300.items():
        _sp(E.exp2(k, 300)["verh_own"], lo, hi, tol=0.06)


def test_k4_zweitmessung_gegen_den_realen_effekt():
    erw = {"Boosting": (1.6, 2.4), "Random Forest": (2.4, 4.7), "Netz": (3.5, 5.0), "lineares Modell": (1.0, 1.0)}
    for k, (lo, hi) in erw.items():
        _sp(E.exp2(k, 3000, True)["verh_real"], lo, hi, tol=0.06)


def test_k4_pd_ueber_die_stuetze_gegen_ale():
    """Abweichung vom realen Effekt (n = 3 000, Mittel über 6 Zufalls-Seeds je Gruppe): PD über die Stütze (`ePr_real`) gegen ALE (`eA_real`), Zählung der Gruppen mit näherer Stütze."""
    erw = {  # (Stütze, ALE, Gruppen mit näherer Stütze) je Messung (erst, zweit)
        "mlp": (((0.027, 0.041), (0.035, 0.055), 5), ((0.029, 0.037), (0.039, 0.049), 5)),
        "gbm": (((0.057, 0.063), (0.026, 0.055), 0), ((0.033, 0.050), (0.027, 0.051), 1)),
        "rf": (((0.036, 0.064), (0.033, 0.047), 1), ((0.031, 0.050), (0.028, 0.053), 3)),
    }
    for kl, je_messung in erw.items():
        for zweit, (st, ale, anzahl) in zip((False, True), je_messung):
            gr = DATA.grid(zweit)["A"][f"{kl}_3000"]
            _sp([x["ePr_real"] for x in gr], *st, tol=T3 + 5e-4)
            _sp([x["eA_real"] for x in gr], *ale, tol=T3 + 5e-4)
            assert sum(1 for x in gr if x["ePr_real"] < x["eA_real"]) == anzahl, (kl, zweit)


def test_k5_unmoegliche_auswertungspunkte_und_lineares_modell_live():
    for k in KL:
        _sp(E.exp2(k, 3000)["inv_ale"], 0.242, 0.251, tol=T3 + 5e-4)
    r = E.linear_live(3000)
    assert r["pd"] == pytest.approx(0.665, abs=T2) and r["je_tag"] == pytest.approx(0.222, abs=0.001) and r["sl_max"] - r["sl_min"] < 1e-9
    assert r["pd"] == pytest.approx(r["ale"], abs=1e-9) == pytest.approx(r["pd_stuetze"], abs=1e-9)


def test_k6_ice_auf_unmoeglichen_zeilen():
    erw = {"Boosting": ((0.57, 0.64), (0.0, 0.0), (4.0, 6.5), (0.955, 0.959)), "Random Forest": ((0.42, 0.45), (0.1, 0.3), (3.1, 3.8), (0.895, 0.902)),
           "Netz": ((0.21, 0.39), (6.0, 30.0), (9.4, 15.3), (0.961, 0.965)), "lineares Modell": ((0.64, 0.65), (0.0, 0.0), (6.6, 7.0), (0.888, 0.889))}
    for k, (ice, flach, feh, r2) in erw.items():
        r = E.exp3(k, 3000)
        _sp(r["ice_imp"], *ice, tol=0.008)
        _sp(r["flach"], *flach, tol=0.6)
        _sp(r["fehler_verh"], *feh, tol=0.06)
        _sp(r["r2"], *r2, tol=T3 + 5e-4)


def test_k7_dosis():
    g = DATA.grid()["C"]
    for dk in ("Boosting", "Netz"):
        r = E.exp4(dk, 0.45)
        z = {x["p0"]: x for x in r["zeilen"]}
        assert [round(E.wahrheiten(p)["pd"], 3) for p in (0.0, 0.2, 0.45, 0.7, 0.9)] == [0.778, 0.623, 0.428, 0.233, 0.078]                 # geschlossene Form, README
        assert [round(z[p]["pd_t"], 3) for p in (0.0, 0.2, 0.45, 0.7, 0.9)] == pytest.approx([0.778, 0.623, 0.428, 0.235, 0.078], abs=0.002)   # eingefrorene Stichprobe (Tabelle der App)
        assert round(z[0.7]["pd_t"], 3) == 0.235
        assert all(z[p]["real"] == pytest.approx(0.778, abs=0.002) for p in z)
    b, n = {x["p0"]: x for x in E.exp4("Boosting", 0.45)["zeilen"]}, {x["p0"]: x for x in E.exp4("Netz", 0.45)["zeilen"]}
    _sp(b[0.0]["eP"], 0.05, 0.13)
    _sp(b[0.0]["eA"], 0.03, 0.07)
    assert b[0.0]["ale_besser_gruppen"] == 3 and n[0.0]["ale_besser_gruppen"] == 0
    _sp(n[0.0]["eP"], 0.03, 0.06)
    _sp(n[0.0]["eA"], 0.04, 0.07)
    _sp(b[0.9]["eP"], 0.15, 0.21)
    _sp(b[0.9]["eA"], 0.11, 0.48)
    assert 5 - b[0.9]["ale_besser_gruppen"] == 4 and n[0.9]["ale_besser_gruppen"] == 4
    _sp(n[0.9]["eP"], 0.31, 0.44)
    _sp(n[0.9]["eA"], 0.16, 0.43)
    _sp(n[0.7]["eP"], 0.28, 0.34)
    _sp(n[0.7]["eA"], 0.02, 0.06)
    assert len(g["gbm_0.9"]) == 5
    for p0 in (0.0, 0.2, 0.45, 0.7, 0.9):                                                              # exakt: PD = (1 − Anteil) · real
        w = E.wahrheiten(p0)
        assert w["verhaeltnis"] == pytest.approx(1 - p0, abs=1e-12)


def test_k8_faecher():
    inv = {5: (0.318, 0.320), 10: (0.268, 0.270), 20: (0.240, 0.251), 50: (0.231, 0.238)}
    for K, (lo, hi) in inv.items():
        _sp(E.exp5("Boosting", K)["inv"], lo, hi, tol=T3 + 5e-4)
    r = E.exp5("Boosting", 20)
    _sp(r["eff0_m"], -0.91, -0.65)
    _sp(r["eff0_t"], 0.04, 0.09)
    _sp(E.exp5("Boosting", 5)["eA_real"], 0.30, 0.36)
    _sp(E.exp5("Boosting", 10)["eA_real"], 0.04, 0.05)
    _sp(E.exp5("Boosting", 5)["hi0"], 2.52, 2.67)
    assert max(E.exp5("Boosting", 10)["hi0"]) < 2.0 and max(E.exp5("Boosting", 5)["hi0"]) > 2.0
    besser = 0
    for zweit in (False, True):                                                                        # K = 10 gegen K = 20, gleiche Gruppen, Boosting: 10 von 10
        b = DATA.grid(zweit)["B"]
        k10, k20 = b[f"{DATA.KLASSEN['Boosting']}_10"], b[f"{DATA.KLASSEN['Boosting']}_20"]
        besser += sum(1 for a, c in zip(k10, k20) if a["eA_real"] < c["eA_real"])
    assert besser == 10


def test_k8_tatsaechliche_faecherzahl_und_anteil_des_ersten_fachs():
    """Neu gerechnet aus dem Generator (kein Modell nötig): die Auswertungszeilen von Teil B (3 000 je Zufalls-Seed, 20 Zufalls-Seeds je Messung, Erst- und Zweitmessung) und die Quantilkanten wie in `ukb_curves.ale_1d`."""
    import ukb_generator as G
    from ukb_feasible import gueltig
    from ukb_rng import SplitMix64
    zahl = {5: [], 10: [], 20: [], 50: []}
    anteil20 = []
    for zweit in (False, True):
        for sd in range(20):
            Xt, _ = G.draw(3000, SplitMix64(4_000_000 + (500000 if zweit else 0) + 2000 + sd))
            for K in zahl:
                e = np.unique(np.quantile(Xt[:, 4], np.linspace(0, 1, K + 1)))
                zahl[K].append(len(e) - 1)
                if K == 20:
                    k = np.clip(np.searchsorted(e, Xt[:, 4], side="left") - 1, 0, len(e) - 2)
                    Zl, Zu = Xt.copy(), Xt.copy()
                    Zl[:, 4], Zu[:, 4] = e[k], e[k + 1]
                    schlecht = np.concatenate([~gueltig(Zl), ~gueltig(Zu)])
                    erstes = np.concatenate([k == 0, k == 0])
                    anteil20.append((schlecht & erstes).sum() / schlecht.sum())
    assert (min(zahl[5]), max(zahl[5])) == (3, 3) and (min(zahl[10]), max(zahl[10])) == (6, 6)
    assert (min(zahl[20]), max(zahl[20])) == (11, 12) and (min(zahl[50]), max(zahl[50])) == (27, 30)
    assert round(100 * min(anteil20)) == 88 and round(100 * max(anteil20)) == 99


def test_zweitmessung_befunde_und_korrekturen():
    z = lambda k, n: E.exp2(k, n, True)
    _sp(z("Boosting", 3000)["verh_own"], 4.1, 7.8, tol=0.06)
    _sp(z("Random Forest", 3000)["verh_own"], 2.7, 5.0, tol=0.06)
    _sp(z("Netz", 3000)["verh_own"], 3.3, 5.3, tol=0.06)
    _sp(z("Boosting", 3000)["sl_pd"], 0.65, 0.73)
    _sp(E.exp3("Netz", 3000, True)["flach"], 12.0, 26.0, tol=0.6)
    b, n = {x["p0"]: x for x in E.exp4("Boosting", 0.45, True)["zeilen"]}, {x["p0"]: x for x in E.exp4("Netz", 0.45, True)["zeilen"]}
    assert b[0.0]["ale_besser_gruppen"] == 4 and n[0.0]["ale_besser_gruppen"] == 0                    # Anteil 0: Boosting ALE näher in 4 von 5, Netz PD näher in 5 von 5
    assert 5 - b[0.9]["ale_besser_gruppen"] == 5 and n[0.9]["ale_besser_gruppen"] == 5               # Anteil 0,9: Boosting ALE schlechter in 5 von 5, Netz besser in 5 von 5
    assert DATA.grid()["meta"]["seeds_je_gruppe"] == {"A": 6, "B": 4, "C": 4} and DATA.grid()["meta"]["gruppen"] == 5
    assert DATA.grid(True)["meta"]["stand"] and DATA.grid(True)["A"]["gbm_3000"] != DATA.grid()["A"]["gbm_3000"]
