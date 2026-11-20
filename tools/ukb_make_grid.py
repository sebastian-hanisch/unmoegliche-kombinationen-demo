"""Raster der Modelle einfrieren (`data/ukb_grid.json`): vier Modellklassen (Boosting, Random Forest, Netz, lineares Modell), PD, ICE und ALE des Vorlaufs gegen die Wahrheiten.

Aufruf (Dev-venv, `OMP_NUM_THREADS=1`; dauert einige zehn Minuten):
    python tools/ukb_make_grid.py             schreibt `data/ukb_grid.json` (Erstmessung)
    python tools/ukb_make_grid.py --zweit     Zweitmessung mit anderen Zufalls-Seeds nach `data/ukb_grid_zweit.json`
    python tools/ukb_make_grid.py --check     rechnet neu und vergleicht (Band) mit der Datei

**Zähleinheit:** Zufalls-Seeds; **Gruppe** = 6 (Teil A) bzw. 4 (Teil B, C) getrennte Zufalls-Seeds, fünf Gruppen je Zelle; je Zufalls-Seed ein Trainingssatz aus dem Generator (`ukb_generator.draw`, SplitMix64) und
3000 (A, B) bzw. 6000 (C) frische Auswertungszeilen. Aussagen als Spanne über die fünf Gruppenwerte.
**Teil A:** vier Klassen, n = 300 und 3000; PD, ALE (K = 20), ICE-Anstieg Vorlauf 2 auf 5 (2000 Zeilen) gegen drei Bezugskurven (PD der wahren Funktion, ALE der wahren Funktion, „real“ = nur Zeilen mit Vorlauf).
**Teil B:** Zahl der Fächer K = 5, 10, 20, 50 (Boosting und Netz, n = 1000) und Wirkung des ersten Fachs. **Teil C:** Anteil ohne Vorlauf p0 = 0, 0,2, 0,45, 0,7, 0,9 (Boosting und Netz, n = 1000)."""

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

import ukb_curves as CV  # noqa: E402
import ukb_generator as G  # noqa: E402
from ukb_rng import SplitMix64  # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "data"
KL = ("gbm", "rf", "mlp", "lin")
NAME = {"gbm": "Boosting", "rf": "Random Forest", "mlp": "Netz", "lin": "lineares Modell"}
GRUPPEN = 5


def modell(kl, seed):
    if kl == "gbm":
        return GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=seed)
    if kl == "rf":
        return RandomForestRegressor(100, min_samples_leaf=3, max_features=0.5, n_jobs=1, random_state=seed)
    if kl == "mlp":
        ct = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [0, 1]), ("sk", StandardScaler(), [2, 3, 4, 5])])
        return make_pipeline(ct, MLPRegressor(hidden_layer_sizes=(16,), alpha=3.0, max_iter=500, solver="lbfgs", random_state=seed))
    ct = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [0, 1]), ("pt", "passthrough", [2, 3, 4, 5])])
    return make_pipeline(ct, LinearRegression())


def r2(pred, ziel):
    return 1.0 - float(np.sum((pred - ziel) ** 2) / np.sum((ziel - ziel.mean()) ** 2))


def seeds_von(basis, teil, zweit):
    return basis + (500000 if zweit else 0) + 1000 * teil


def job_a(args):
    kl, n, sd, zweit = args
    X, y = G.draw(n, SplitMix64(seeds_von(1_000_000, 1, zweit) + sd))
    Xt, _ = G.draw(3000, SplitMix64(seeds_von(2_000_000, 1, zweit) + sd))
    real = np.isin(Xt[:, 0], (1, 2))
    pd_t = CV.pd_kurve(G.wahrheit, Xt, 4)
    e, a, _, _ = CV.ale_1d(G.wahrheit, Xt, 4)
    ale_t = np.interp(CV.GITTER, e, a)
    re = CV.pd_kurve(G.wahrheit, Xt[real], 4)
    m = modell(kl, sd).fit(X, y)
    f = m.predict
    pd_m, pdr_m = CV.pd_kurve(f, Xt, 4), CV.pd_kurve(f, Xt[real], 4)
    ale_m, inv, _, _ = CV.ale_an_gitter(f, Xt)
    Xi = Xt[:2000]
    ri = np.isin(Xi[:, 0], (1, 2))
    sl = CV.ice_anstiege(f, Xi, 4)
    V5 = Xt.copy()
    V5[:, 4] = 5.0
    dv = np.abs(f(V5) - G.wahrheit(V5))
    d = CV.anstieg
    return dict(kl=kl, n=n, sd=sd, pd_inc=d(pd_m).tolist(), ale_inc=d(ale_m).tolist(), sl_pd=d(pd_m)[2], sl_ale=d(ale_m)[2], sl_pdr=d(pdr_m)[2],
                eP_own=np.abs(d(pd_m) - d(pd_t)).mean(), eA_own=np.abs(d(ale_m) - d(ale_t)).mean(), eP_real=np.abs(d(pd_m) - d(re)).mean(), eA_real=np.abs(d(ale_m) - d(re)).mean(),
                ePr_real=np.abs(d(pdr_m) - d(re)).mean(), inv_ale=inv, ice_imp=np.abs(sl[~ri]).mean(), ice_pos=np.abs(sl[ri]).mean(), flat_imp=(np.abs(sl[~ri]) < 0.1).mean(),
                err_imp=dv[~real].mean(), err_pos=dv[real].mean(), r2mu=r2(f(Xt), G.wahrheit(Xt)), t_sl_pd=d(pd_t)[2], t_sl_ale=d(ale_t)[2], t_sl_real=d(re)[2])


def job_b(args):
    kl, sd, zweit = args
    X, y = G.draw(1000, SplitMix64(seeds_von(3_000_000, 2, zweit) + sd))
    Xt, _ = G.draw(3000, SplitMix64(seeds_von(4_000_000, 2, zweit) + sd))
    real = np.isin(Xt[:, 0], (1, 2))
    re = CV.pd_kurve(G.wahrheit, Xt[real], 4)
    m = modell(kl, sd).fit(X, y)
    out = []
    for K in (5, 10, 20, 50):
        e, a, inv, eff = CV.ale_1d(m.predict, Xt, 4, K)
        et, at, _, efft = CV.ale_1d(G.wahrheit, Xt, 4, K)
        ale_m, ale_t = np.interp(CV.GITTER, e, a), np.interp(CV.GITTER, et, at)
        d = CV.anstieg
        out.append(dict(kl=kl, K=K, sd=sd, eA_real=np.abs(d(ale_m) - d(re)).mean(), eA_own=np.abs(d(ale_m) - d(ale_t)).mean(), inv=inv, eff0_m=eff[0], eff0_t=efft[0], hi0=e[1]))
    return out


def job_c(args):
    kl, p0, sd, zweit = args
    X, y = G.draw(1000, SplitMix64(seeds_von(5_000_000, 3, zweit) + sd), p0)
    Xt, _ = G.draw(6000, SplitMix64(seeds_von(6_000_000, 3, zweit) + sd), p0)
    real = np.isin(Xt[:, 0], (1, 2))
    pd_t, re = CV.pd_kurve(G.wahrheit, Xt, 4), CV.pd_kurve(G.wahrheit, Xt[real], 4)
    m = modell(kl, sd).fit(X, y)
    pd_m = CV.pd_kurve(m.predict, Xt, 4)
    ale_m, _, _, _ = CV.ale_an_gitter(m.predict, Xt)
    d = CV.anstieg
    return dict(kl=kl, p0=p0, sd=sd, inv=1 - real.mean(), t_sl_pd=d(pd_t)[2], t_sl_real=d(re)[2], sl_pd=d(pd_m)[2], sl_ale=d(ale_m)[2],
                eP_real=np.abs(d(pd_m) - d(re)).mean(), eA_real=np.abs(d(ale_m) - d(re)).mean())


def gruppen(zeilen, n_je_gruppe, felder):
    """Gruppenmittel: Liste (je Gruppe) von Wörterbüchern; `zeilen` nach Zufalls-Seed sortiert (sd = Gruppe * n_je_gruppe + k)."""
    out = []
    for g in range(GRUPPEN):
        z = [r for r in zeilen if g * n_je_gruppe <= r["sd"] < (g + 1) * n_je_gruppe]
        assert len(z) == n_je_gruppe
        out.append({k: float(np.mean([r[k] for r in z])) for k in felder})
    return out


def aggregiere_a(zeilen):
    felder = ["sl_pd", "sl_ale", "sl_pdr", "eP_own", "eA_own", "eP_real", "eA_real", "ePr_real", "inv_ale", "ice_imp", "ice_pos", "flat_imp", "err_imp", "err_pos", "r2mu", "t_sl_pd", "t_sl_ale", "t_sl_real"]
    out = {}
    for kl in KL:
        for n in (300, 3000):
            z = sorted([r for r in zeilen if r["kl"] == kl and r["n"] == n], key=lambda r: r["sd"])
            gr = gruppen(z, 6, felder)
            for g, grp in enumerate(gr):
                zz = z[g * 6:(g + 1) * 6]
                grp["ale_naeher_own"] = float(np.mean([r["eA_own"] < r["eP_own"] for r in zz]))
                grp["ale_naeher_real"] = float(np.mean([r["eA_real"] < r["eP_real"] for r in zz]))
                grp["pdr_naeher_real"] = float(np.mean([r["ePr_real"] < r["eA_real"] for r in zz]))
                grp["pd_inc"] = np.mean([r["pd_inc"] for r in zz], axis=0).tolist()
                grp["ale_inc"] = np.mean([r["ale_inc"] for r in zz], axis=0).tolist()
            out[f"{kl}_{n}"] = gr
    return out


def rechne(zweit=False):
    with ProcessPoolExecutor(max_workers=4) as ex:
        a = list(ex.map(job_a, [(kl, n, sd, zweit) for kl in KL for n in (300, 3000) for sd in range(6 * GRUPPEN)]))
        b = [r for rs in ex.map(job_b, [(kl, sd, zweit) for kl in ("gbm", "mlp") for sd in range(4 * GRUPPEN)]) for r in rs]
        c = list(ex.map(job_c, [(kl, p0, sd, zweit) for kl in ("gbm", "mlp") for p0 in (0.0, 0.2, 0.45, 0.7, 0.9) for sd in range(4 * GRUPPEN)]))
    out = {"A": aggregiere_a(a), "B": {}, "C": {}}
    for kl in ("gbm", "mlp"):
        for K in (5, 10, 20, 50):
            z = sorted([r for r in b if r["kl"] == kl and r["K"] == K], key=lambda r: r["sd"])
            out["B"][f"{kl}_{K}"] = gruppen(z, 4, ["eA_real", "eA_own", "inv", "eff0_m", "eff0_t", "hi0"])
        for p0 in (0.0, 0.2, 0.45, 0.7, 0.9):
            z = sorted([r for r in c if r["kl"] == kl and r["p0"] == p0], key=lambda r: r["sd"])
            out["C"][f"{kl}_{p0}"] = gruppen(z, 4, ["inv", "t_sl_pd", "t_sl_real", "sl_pd", "sl_ale", "eP_real", "eA_real"])
            for g, grp in enumerate(out["C"][f"{kl}_{p0}"]):
                zz = z[g * 4:(g + 1) * 4]
                grp["ale_naeher_real"] = float(np.mean([r["eA_real"] < r["eP_real"] for r in zz]))
    return out


def runde(x, st=6):
    if isinstance(x, float):
        return float(f"{x:.3g}") if x != 0 and abs(x) < 10.0 ** -(st - 1) else round(x, st)
    if isinstance(x, dict):
        return {k: runde(v, st) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [runde(v, st) for v in x]
    return x


def vergleiche(neu, alt, pfad="", fehler=None):
    fehler = [] if fehler is None else fehler
    if isinstance(alt, dict):
        for k in alt:
            if k != "meta":
                vergleiche(neu[k], alt[k], f"{pfad}/{k}", fehler)
    elif isinstance(alt, list):
        for i, (a, b) in enumerate(zip(alt, neu)):
            vergleiche(b, a, f"{pfad}[{i}]", fehler)
    elif isinstance(alt, float) and abs(neu - alt) > 1e-4 * max(1.0, abs(alt)) + 2e-6:
        fehler.append((pfad, neu, alt))
    return fehler


if __name__ == "__main__":
    import datetime

    import sklearn

    ap = argparse.ArgumentParser()
    ap.add_argument("--zweit", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = runde(rechne(a.zweit))
    datei = DATA / ("ukb_grid_zweit.json" if a.zweit else "ukb_grid.json")
    if a.check:
        fe = vergleiche(json.loads(json.dumps(out)), json.loads(datei.read_text(encoding="utf-8")))
        print("CHECK:", "gleich" if not fe else f"{len(fe)} Abweichungen, erste: {fe[:3]}")
        sys.exit(1 if fe else 0)
    meta = {"quelle": "Erfundener Generator (verweilzeit-demo, kopiert, SplitMix64); scikit-learn-Modelle, eigene PD, ICE, ALE (ukb_curves)", "stand": datetime.date.today().isoformat(),
            "scikit-learn": sklearn.__version__, "numpy": np.__version__, "gruppen": GRUPPEN, "seeds_je_gruppe": {"A": 6, "B": 4, "C": 4},
            "hinweis": "Eingefrorene Messung (Dev-venv); Neu-Rechnen und Abgleich mit `--check`."}
    datei.write_text(json.dumps({"meta": meta, **out}, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print("geschrieben:", datei)
