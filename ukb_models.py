"""Lineares Modell live (kleinste Quadrate in numpy, kein Paket): Art und Kunde als Indikatoren, dazu Zoll, Wochenende, Vorlauf, Gewicht, ein Achsenabschnitt.
Die anderen drei Modellklassen (Boosting, Random Forest, Netz) sind eingefroren (`data/ukb_grid.json`, `tools/ukb_make_grid.py`)."""

import numpy as np


def _entwurf(X):
    art = np.eye(4)[X[:, 0].astype(int)]
    kunde = np.eye(12)[X[:, 1].astype(int)]
    return np.column_stack([np.ones(len(X)), art[:, 1:], kunde[:, 1:], X[:, 2], X[:, 3], X[:, 4], X[:, 5]])


def fit_linear(X, y):
    """Kleinste Quadrate; gibt `predict(Z)` zurück."""
    beta = np.linalg.lstsq(_entwurf(X), y, rcond=None)[0]
    return lambda Z: _entwurf(np.asarray(Z, float)) @ beta
