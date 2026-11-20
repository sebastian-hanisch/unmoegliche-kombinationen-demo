"""PD, ICE und ALE (eigene Implementierungen, kein Paket) und die Wahrheitskurven der bekannten Funktion in geschlossener Form.

* **Partielle Abhängigkeit (PD):** mittelt die Vorhersage über die Randverteilung der übrigen Merkmale, indem der Vorlauf für **alle** Zeilen auf den Gitterwert gesetzt wird (Friedman 2001; Definition in Goldstein u. a. 2015 und Apley/Zhu 2020).
* **ICE:** dieselbe Kurve je Zeile; die PD ist ihr Mittel (Goldstein u. a. 2015, S. 5). **ALE:** Summe bedingter lokaler Differenzen über Fächer der Quantile (Apley/Zhu 2020).
* **Wahrheit je Funktional:** die PD der wahren Funktion μ, die ALE von μ (Integral der bedingten Ableitung, Apley/Zhu 2020 Satz 1) und der **reale Effekt** (Mittel von μ über die Zeilen, in denen der Vorlauf vorkommt) sind drei verschiedene Zahlen.
Kurven werden relativ zum ersten Gitterpunkt (Vorlauf 2) angegeben (`anstieg`)."""

import math
from fractions import Fraction

import numpy as np

import ukb_generator as G
from ukb_feasible import gueltig

GITTER = np.array([2.0, 3.0, 4.0, 5.0, 6.0])


def pd_kurve(f, X, j, gitter=GITTER):
    out = []
    for g in gitter:
        Z = X.copy()
        Z[:, j] = g
        out.append(float(np.mean(f(Z))))
    return np.array(out)


def ice_anstiege(f, X, j, von=2.0, bis=5.0):
    """Anstieg der ICE-Kurve jeder Zeile von `von` auf `bis` (in ln-Verweilzeit; kein Anstieg je Tag)."""
    hi, lo = X.copy(), X.copy()
    hi[:, j], lo[:, j] = bis, von
    return f(hi) - f(lo)


def ale_1d(f, X, j, K=20, edges=None):
    """(Kanten, zentrierte ALE an den Kanten, Anteil unmöglicher Auswertungspunkte, Effekt je Fach). Kanten: Quantile von X[:, j] oder die übergebenen `edges`. Das erste Fach enthält alle Zeilen mit dem kleinsten Wert (hier Vorlauf 0)."""
    edges = np.unique(np.quantile(X[:, j], np.linspace(0, 1, K + 1))) if edges is None else np.asarray(edges, float)
    Kk = len(edges) - 1
    if Kk < 1:
        raise ValueError("ALE nicht definiert: die Spalte hat keine Variation (weniger als zwei verschiedene Kanten)")
    k = np.clip(np.searchsorted(edges, X[:, j], side="left") - 1, 0, Kk - 1)
    Zl, Zu = X.copy(), X.copy()
    Zl[:, j], Zu[:, j] = edges[k], edges[k + 1]
    delta = f(Zu) - f(Zl)
    cnt = np.bincount(k, minlength=Kk)
    eff = np.array([delta[k == i].mean() if cnt[i] else 0.0 for i in range(Kk)])
    acc = np.concatenate([[0.0], np.cumsum(eff)])
    mid = (acc[:-1] + acc[1:]) / 2
    acc = acc - (mid * cnt).sum() / cnt.sum()
    inv = 1.0 - float(np.mean(np.concatenate([gueltig(Zl), gueltig(Zu)])))
    return edges, acc, inv, eff


def anstieg(kurve):
    """Anstieg relativ zum ersten Gitterpunkt, an den Punkten 3 bis 6."""
    kurve = np.asarray(kurve)
    return kurve[1:] - kurve[0]


def ale_an_gitter(f, X, j=4, K=20, gitter=GITTER):
    e, a, inv, eff = ale_1d(f, X, j, K)
    return np.interp(gitter, e, a), inv, eff, e


# ------------------------------------------------------------------------------------------------ Wahrheit in geschlossener Form
def _teil(g):
    """ln(Median) der wahren Funktion bei Vorlauf g für Export und Umschlag (ohne Kundeneffekt)."""
    return math.log(0.6 + 0.85 * g), math.log(0.3 + 0.8 * g)


def pd_wahrheit(gitter=GITTER, p0=G.P0_STANDARD):
    """PD der wahren Funktion relativ zum ersten Gitterpunkt: nur Export und Umschlag tragen bei (Import und Leer sind flach); Gewicht = ihre Anteile."""
    pi, pe, pu, pl = G.art_wahrscheinlichkeiten(p0)
    e0, u0 = _teil(gitter[0])
    return np.array([float(pe) * (_teil(g)[0] - e0) + float(pu) * (_teil(g)[1] - u0) for g in gitter])


def real_wahrheit(gitter=GITTER, p0=G.P0_STANDARD):
    """Realer Effekt: Mittel von μ über die Zeilen, in denen der Vorlauf vorkommt (Export, Umschlag), relativ zum ersten Gitterpunkt. Gleich PD / (1 − p0): exakt."""
    return pd_wahrheit(gitter, p0) / (1 - p0)


def ale_wahrheit(gitter=GITTER, p0=G.P0_STANDARD):
    """ALE der wahren Funktion relativ zum ersten Gitterpunkt: Integral der bedingten Ableitung nach dem Vorlauf (Gauß-Legendre, 64 Punkte je Intervall).
    An der Stelle t gewichtet die Dichte jeder Art (Export 1/9 auf [1, 10], Umschlag 1/5,5 auf [0,5; 6], mal ihr Anteil) die Ableitung; p0 kürzt sich heraus."""
    pi, pe, pu, pl = (float(x) for x in G.art_wahrscheinlichkeiten(p0))
    x_, w_ = np.polynomial.legendre.leggauss(64)

    def ableitung(t):
        de = pe / 9.0 if 1 <= t <= 10 else 0.0
        du = pu / 5.5 if 0.5 <= t <= 6 else 0.0
        return (de * 0.85 / (0.6 + 0.85 * t) + du * 0.8 / (0.3 + 0.8 * t)) / (de + du)

    out = [0.0]
    for lo, hi in zip(gitter[:-1], gitter[1:]):
        t = 0.5 * (hi - lo) * x_ + 0.5 * (hi + lo)
        out.append(out[-1] + 0.5 * (hi - lo) * float(sum(w * ableitung(tt) for w, tt in zip(w_, t))))
    return np.array(out)


def wahrheiten_anstieg_2_5(p0=G.P0_STANDARD):
    """(PD, real, ALE) der wahren Funktion: Anstieg von Vorlauf 2 auf 5."""
    return float(pd_wahrheit(p0=p0)[3]), float(real_wahrheit(p0=p0)[3]), float(ale_wahrheit(p0=p0)[3])
