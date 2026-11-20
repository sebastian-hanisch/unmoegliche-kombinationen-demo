"""Erfundener Container-Generator (Verweilzeit): KOPIE der Formeln und Anteile aus `verweilzeit-demo` (`vwz_generator.py`, `vwz_constants.py`), kein Import.
Alles hier ist ausgedacht (kein Echtdatenbezug). Neu gegenüber der Quelle: der Anteil der Container ohne Vorlauf (Import plus Leer) ist ein Regler `p0` (Standard 0,45),
und gezogen wird mit dem eigenen Zufallsstrom SplitMix64 (`ukb_rng`), auf jeder Plattform bitgleich. `tests/test_generator_kopie.py` prüft die Formeln gegen eingefrorene Werte der Quelle.

Merkmale (Spalten): 0 art (0 Import, 1 Export, 2 Umschlag, 3 Leer), 1 kunde (0..11), 2 zoll, 3 wochenende, 4 vorlauf, 5 gewicht (reines Rauschen).
Der Vorlauf ist bei Import und Leer immer 0, bei Export gleichverteilt über [1, 10], bei Umschlag über [0,5; 6]; Zoll gibt es nur bei Import. Die bekannte Funktion μ(x) ist die wahre Mitte der Log-Verweilzeit."""

import math
from fractions import Fraction

import numpy as np

from ukb_rng import SplitMix64

ART_LABELS = ("Import", "Export", "Umschlag", "Leer")
ART_P = (0.35, 0.30, 0.25, 0.10)             # Anteile bei p0 = 0,45
P0_STANDARD = 0.45
K_KUNDEN = 12
KUNDEN_SEED = 12345
U_SD = 0.30
U_SCALE = (1.0, 0.3, 0.3, 1.0)
S_BASE = (0.55, 0.25, 0.35, 0.90)
P_ZOLL = 0.25
NAMEN = ("Art", "Kunde", "Zoll", "Wochenende", "Vorlauf", "Gewicht")
UK = np.random.default_rng(KUNDEN_SEED).normal(0.0, U_SD, K_KUNDEN)


def art_wahrscheinlichkeiten(p0):
    """Anteile (Import, Export, Umschlag, Leer) bei Anteil `p0` ohne Vorlauf: Import : Leer = 0,35 : 0,10, Export : Umschlag = 0,30 : 0,25 (bei p0 = 0,45 die Anteile der Quelle). Brüche bleiben Brüche."""
    return (p0 * Fraction(7, 9) if isinstance(p0, Fraction) else p0 * 0.35 / 0.45,
            (1 - p0) * Fraction(6, 11) if isinstance(p0, Fraction) else (1 - p0) * 0.30 / 0.55,
            (1 - p0) * Fraction(5, 11) if isinstance(p0, Fraction) else (1 - p0) * 0.25 / 0.55,
            p0 * Fraction(2, 9) if isinstance(p0, Fraction) else p0 * 0.10 / 0.45)


def log_median_and_s(art, kunde, zoll, we, vorlauf):
    """Bekannte bedingte Verteilung: log-Verweilzeit ~ N(mu, s^2). Vektorisiert."""
    art = np.asarray(art)
    u = UK[kunde] * np.take(U_SCALE, art)
    med = np.select([art == 0, art == 1, art == 2], [3.5 * np.exp(0.7 * zoll + 0.30 * we), 0.6 + 0.85 * vorlauf, 0.3 + 0.8 * vorlauf], default=7.0 * np.exp(0.30 * we))
    return np.log(med) + u, np.take(S_BASE, art)


def wahrheit(X):
    """Die bekannte Funktion μ(x) auf den Spalten 0 bis 4."""
    mu, _ = log_median_and_s(X[:, 0].astype(int), X[:, 1].astype(int), X[:, 2], X[:, 3], X[:, 4])
    return mu


def _normal(rng: SplitMix64) -> float:
    """Box-Muller mit math.log/cos (schnell; die Gleichverteilten sind bitgleich, die Normalzahlen können sich unter Linux in der letzten Binärstelle unterscheiden)."""
    return math.sqrt(-2.0 * math.log(max(rng.random(), 1e-12))) * math.cos(2.0 * math.pi * rng.random())


def draw(n: int, rng: SplitMix64, p0: float = P0_STANDARD):
    """(X (n, 6), log-Verweilzeit (n,)): n Container aus dem Generator mit Anteil `p0` ohne Vorlauf."""
    pa = art_wahrscheinlichkeiten(p0)
    cum = np.cumsum(pa)
    X = np.zeros((n, 6))
    for i in range(n):
        u = rng.random()
        art = min(int(np.searchsorted(cum, u)), 3)
        X[i, 0] = art
        X[i, 1] = rng.below(K_KUNDEN)
        X[i, 2] = 1.0 if (art == 0 and rng.random() < P_ZOLL) else 0.0
        X[i, 3] = 1.0 if rng.random() < 2 / 7 else 0.0
        v = rng.random()
        X[i, 4] = 1 + 9 * v if art == 1 else (0.5 + 5.5 * v if art == 2 else 0.0)
        X[i, 5] = _normal(rng)
    mu, s = log_median_and_s(X[:, 0].astype(int), X[:, 1].astype(int), X[:, 2], X[:, 3], X[:, 4])
    z = np.array([_normal(rng) for _ in range(n)])
    return X, mu + s * z
