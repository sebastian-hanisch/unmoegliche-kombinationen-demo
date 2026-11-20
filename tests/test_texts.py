"""Zwei Fassungen desselben Textes (App und README) werden verglichen: die Tabelle „Wo die Annahmen enden“ steht wörtlich in beiden. Dazu: Aufbau der eingefrorenen Daten und Formelblöcke."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import ukb_data as DATA
import ukb_texts as TX

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_grenzen_tabelle_steht_wortgleich_in_der_readme_und_in_der_app():
    assert TX.GRENZEN_MARKDOWN.strip() in README
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception
    assert any(TX.GRENZEN_MARKDOWN.strip() == m.value.strip() for m in at.markdown)
    assert any("📐 Mathematische Formulierung" == e.label for e in at.expander)           # Zugriff über den Titel, nicht über den Index


def test_formeln_haben_kein_ungeschuetztes_prozentzeichen():
    for _, formel in TX.MATHE_BLOECKE:
        assert "%" not in formel


def test_daten_aufbau_und_meta():
    for zweit in (False, True):
        g = DATA.grid(zweit)
        assert g["meta"]["quelle"] and len(g["meta"]["stand"]) == 10 and g["meta"]["gruppen"] == 5
        assert sorted(g["A"]) == sorted(f"{k}_{n}" for k in ("gbm", "rf", "mlp", "lin") for n in (300, 3000))
        assert all(len(v) == 5 for v in g["A"].values()) and all(len(v) == 5 for v in g["B"].values()) and all(len(v) == 5 for v in g["C"].values())
        assert sorted(g["B"]) == sorted(f"{k}_{K}" for k in ("gbm", "mlp") for K in (5, 10, 20, 50))
        assert sorted(g["C"]) == sorted(f"{k}_{p}" for k in ("gbm", "mlp") for p in (0.0, 0.2, 0.45, 0.7, 0.9))
    a, b = DATA.grid(), DATA.grid(True)
    assert a["A"]["gbm_3000"] != b["A"]["gbm_3000"]                                       # andere Zufalls-Seeds, andere Zahlen
    for gr in a["A"]["lin_3000"]:
        assert abs(gr["sl_pd"] - gr["sl_ale"]) < 1e-5                                      # lineares Modell: PD = ALE
        assert len(gr["pd_inc"]) == len(gr["ale_inc"]) == 4
