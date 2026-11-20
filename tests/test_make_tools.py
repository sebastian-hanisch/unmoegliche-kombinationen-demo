"""`tools/ukb_make_grid.py` und die eingefrorene Datei: Neu-Rechnen eines Seeds als **Band** (scikit-learn und BLAS können in der letzten Stelle abweichen). Ganze Läufe (`--check`) sind zu teuer
für die Suite und stehen in der README."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import ukb_make_grid as MG  # noqa: E402

import ukb_data as DATA  # noqa: E402


def test_ein_zufalls_seed_der_gruppe_neu_gerechnet_liegt_im_gruppenband():
    """Linear, n = 300, Zufalls-Seed 0 (Gruppe 1): der Anstieg in der PD liegt im Band der eingefrorenen Gruppenmittel (kleinster bis größter Wert, Toleranz 0,06: ein Seed streut mehr als ein Gruppenmittel)."""
    r = MG.job_a(("lin", 300, 0, False))
    gr = DATA.grid()["A"]["lin_300"]
    assert min(g["sl_pd"] for g in gr) - 0.06 <= r["sl_pd"] <= max(g["sl_pd"] for g in gr) + 0.06
    assert r["sl_pd"] == pytest.approx(r["sl_ale"], abs=1e-6)                         # lineares Modell: PD = ALE
    assert r["t_sl_pd"] == pytest.approx(0.43, abs=0.03) and r["t_sl_real"] == pytest.approx(0.78, abs=0.03)


def test_gruppen_aggregation_und_rundung():
    zeilen = [{"sd": i, "x": float(i)} for i in range(30)]
    gr = MG.gruppen(zeilen, 6, ["x"])
    assert [g["x"] for g in gr] == [2.5, 8.5, 14.5, 20.5, 26.5]
    assert MG.runde({"a": 0.123456789, "b": 3e-14, "c": [1.0, 2]}) == {"a": 0.123457, "b": 3e-14, "c": [1.0, 2]}


def test_vergleiche_findet_abweichung_und_ignoriert_meta():
    assert MG.vergleiche({"x": 1.0, "meta": 5}, {"x": 1.0, "meta": 6}) == []
    assert len(MG.vergleiche({"x": 1.1}, {"x": 1.0})) == 1 and np.isclose(1.0, 1.0)
