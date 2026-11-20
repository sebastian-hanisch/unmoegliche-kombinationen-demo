"""Eingefrorene Messungen laden (`data/ukb_grid.json`, geschrieben von `tools/ukb_make_grid.py`; Quelle und Stand im Feld `meta`). Die App rechnet nur mit diesen Zahlen
und den eigenen Kurven (`ukb_curves`); scikit-learn kommt zur Laufzeit nicht vor."""

import functools
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
KLASSEN = {"Boosting": "gbm", "Random Forest": "rf", "Netz": "mlp", "lineares Modell": "lin"}
KLASSE_NAME = {v: k for k, v in KLASSEN.items()}


@functools.lru_cache(maxsize=None)
def grid(zweit: bool = False) -> dict:
    return json.loads((DATA / ("ukb_grid_zweit.json" if zweit else "ukb_grid.json")).read_text(encoding="utf-8"))
