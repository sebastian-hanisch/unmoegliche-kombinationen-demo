"""Rechnungen der fünf Experimente (rein, ohne Streamlit). Zähleinheiten: Zeilen (Anteile unmöglicher Zeilen), Zufalls-Seeds (je Zufalls-Seed ein Trainingssatz und frische Auswertungszeilen),
Auswertungspunkte der ALE (Kanten der Fächer), Gruppen (6 bzw. 4 getrennte Zufalls-Seeds; Spannen über fünf Gruppen, nie ein Punktwert aus einer festen Ziehung)."""

import functools
from fractions import Fraction

import numpy as np

import ukb_curves as CV
import ukb_data as DATA
import ukb_feasible as FE
import ukb_generator as G
import ukb_models as MOD
from ukb_rng import SplitMix64

SETZ = {"vorlauf": 4, "art": 0, "zoll": 2}


def parse_wert(label: str):
    """'Vorlauf = 0,5' → ('vorlauf', 0.5); 'Art = Export' → ('art', 1); 'Zoll = 1' → ('zoll', 1)."""
    name, wert = (t.strip() for t in label.split("="))
    if name == "Art":
        return "art", G.ART_LABELS.index(wert)
    if name == "Vorlauf":
        return "vorlauf", Fraction(wert.replace(",", "."))
    return "zoll", int(wert)


def p0_bruch(p0: float) -> Fraction:
    return Fraction(round(p0 * 20), 20)


@functools.lru_cache(maxsize=None)
def stichprobe(p0: float, n: int = 6000):
    X, _ = G.draw(n, SplitMix64(9000 + round(p0 * 100)), p0)
    return X


def exp1(label: str, p0: float) -> dict:
    """Anteil unmöglicher Zeilen, wenn ein Merkmal für alle Zeilen gesetzt wird: exakt (Brüche), je Art, und gezählt an einer Stichprobe von 6 000 Zeilen."""
    merkmal, wert = parse_wert(label)
    pb = p0_bruch(p0)
    je_art = FE.unmoeglich_je_art(merkmal, wert, pb)
    exakt = FE.unmoeglich_anteil(merkmal, wert, pb)
    X = stichprobe(p0).copy()
    X[:, SETZ[merkmal]] = float(wert)
    anteil = 1.0 - float(FE.gueltig(X).mean())
    se = float(np.sqrt(float(exakt) * (1 - float(exakt)) / len(X)))
    return {"merkmal": merkmal, "wert": wert, "p0": pb, "exakt": exakt, "je_art": je_art, "stichprobe": anteil, "se": se, "n": len(X)}


def wahrheiten(p0: float = G.P0_STANDARD) -> dict:
    pd, real, ale = CV.wahrheiten_anstieg_2_5(p0)
    return {"pd": pd, "real": real, "ale": ale, "verhaeltnis": pd / real, "p0": p0}


def _spanne(werte):
    return (float(min(werte)), float(max(werte)))


def exp2(klasse: str, n: int, zweit: bool = False) -> dict:
    """Vier Modellklassen, drei Bezugskurven: Anstiege Vorlauf 2 auf 5 und Kurvenabweichungen (Spannen über fünf Gruppen zu 6 Zufalls-Seeds)."""
    gr = DATA.grid(zweit)["A"][f"{DATA.KLASSEN[klasse]}_{n}"]
    pd_t, real_t, ale_t = (float(np.mean([g[k] for g in gr])) for k in ("t_sl_pd", "t_sl_real", "t_sl_ale"))
    zwischen = sum(1 for g in gr if min(pd_t, real_t) - 1e-9 <= g["sl_pd"] <= max(pd_t, real_t) + 1e-9)
    return {"klasse": klasse, "n": n, "n_gruppen": len(gr), "sl_pd": _spanne([g["sl_pd"] for g in gr]), "sl_ale": _spanne([g["sl_ale"] for g in gr]),
            "sl_pdr": _spanne([g["sl_pdr"] for g in gr]), "wahr_pd": pd_t, "wahr_real": real_t, "wahr_ale": ale_t, "zwischen": zwischen,
            "verh_own": _spanne([g["eP_own"] / g["eA_own"] for g in gr]), "verh_real": _spanne([g["eP_real"] / g["eA_real"] for g in gr]),
            "naeher_own": _spanne([g["ale_naeher_own"] for g in gr]), "naeher_real": _spanne([g["ale_naeher_real"] for g in gr]),
            "pdr_naeher": _spanne([g["pdr_naeher_real"] for g in gr]), "pdr_abw": float(np.mean([g["ePr_real"] for g in gr])), "ale_abw": float(np.mean([g["eA_real"] for g in gr])),
            "inv_ale": _spanne([g["inv_ale"] for g in gr]), "pd_inc": np.mean([g["pd_inc"] for g in gr], axis=0).tolist(), "ale_inc": np.mean([g["ale_inc"] for g in gr], axis=0).tolist()}


def exp3(klasse: str, n: int, zweit: bool = False) -> dict:
    """ICE auf unmöglichen Zeilen (Import und Leer): mittlerer Betrag des Anstiegs Vorlauf 2 auf 5 (Wahrheit 0), Anteil flacher Kurven, Fehler bei Vorlauf 5, R² gegen μ."""
    gr = DATA.grid(zweit)["A"][f"{DATA.KLASSEN[klasse]}_{n}"]
    return {"klasse": klasse, "n": n, "n_gruppen": len(gr), "ice_imp": _spanne([g["ice_imp"] for g in gr]), "ice_pos": _spanne([g["ice_pos"] for g in gr]),
            "flach": _spanne([100 * g["flat_imp"] for g in gr]), "fehler_verh": _spanne([g["err_imp"] / g["err_pos"] for g in gr]), "r2": _spanne([g["r2mu"] for g in gr])}


@functools.lru_cache(maxsize=None)
def linear_live(n: int, seed: int = 31) -> dict:
    """Das lineare Modell live (kleinste Quadrate auf n erzeugten Zeilen): ICE-Anstiege Vorlauf 2 auf 5 aller 3 000 Auswertungszeilen und PD, ALE, PD über die Stütze."""
    X, y = G.draw(n, SplitMix64(seed))
    Xt, _ = G.draw(3000, SplitMix64(seed + 1))
    f = MOD.fit_linear(X, y)
    sl = CV.ice_anstiege(f, Xt, 4)
    real = np.isin(Xt[:, 0], (1, 2))
    return {"n": n, "sl_min": float(sl.min()), "sl_max": float(sl.max()), "pd": float(CV.anstieg(CV.pd_kurve(f, Xt, 4))[2]), "ale": float(CV.anstieg(CV.ale_an_gitter(f, Xt)[0])[2]),
            "pd_stuetze": float(CV.anstieg(CV.pd_kurve(f, Xt[real], 4))[2]), "je_tag": float(sl.mean() / 3)}


def exp4(dk: str, p0: float, zweit: bool = False) -> dict:
    """Dosis: Anteil ohne Vorlauf. Je Raster-p0 (Boosting oder Netz, n = 1000, fünf Gruppen zu 4 Zufalls-Seeds) und live die geschlossene Wahrheit beim gewählten p0."""
    g = DATA.grid(zweit)["C"]
    zeilen = []
    for q in (0.0, 0.2, 0.45, 0.7, 0.9):
        gr = g[f"{DATA.KLASSEN[dk]}_{q}"]
        zeilen.append({"p0": q, "inv": float(np.mean([x["inv"] for x in gr])), "pd_t": float(np.mean([x["t_sl_pd"] for x in gr])), "real": float(np.mean([x["t_sl_real"] for x in gr])),
                       "eP": _spanne([x["eP_real"] for x in gr]), "eA": _spanne([x["eA_real"] for x in gr]), "ale_naeher": sum(1 for x in gr if x["ale_naeher_real"] > 0.5),
                       "ale_besser_gruppen": sum(1 for x in gr if x["eA_real"] < x["eP_real"]), "pd_m": _spanne([x["sl_pd"] for x in gr]), "ale_m": _spanne([x["sl_ale"] for x in gr])})
    w = wahrheiten(p0)
    return {"klasse": dk, "zeilen": zeilen, "live": w}


def exp5(dk: str, K: int, zweit: bool = False) -> dict:
    """ALE ist nicht gratis: Zahl der Fächer K (n = 1000, fünf Gruppen zu 4 Zufalls-Seeds), Anteil unmöglicher Auswertungspunkte, erstes Fach, Abweichung gegen den realen Effekt."""
    gr = DATA.grid(zweit)["B"][f"{DATA.KLASSEN[dk]}_{K}"]
    return {"klasse": dk, "K": K, "inv": _spanne([x["inv"] for x in gr]), "eA_real": _spanne([x["eA_real"] for x in gr]), "eff0_m": _spanne([x["eff0_m"] for x in gr]),
            "eff0_t": _spanne([x["eff0_t"] for x in gr]), "hi0": _spanne([x["hi0"] for x in gr]), "n_gruppen": len(gr)}
