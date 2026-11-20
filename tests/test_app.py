"""End-to-end Smoke-Test via AppTest: Standardlauf, jeder Button, jeder Regler an den Grenzen, jede Auswahl; kein Python-Fehler (insbesondere keine doppelten Diagramm-Schlüssel)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import ukb_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at):
    return " ".join(x.value for art in (at.success, at.info, at.warning) for x in art)


def test_app_loads_and_shows_six_verdicts():
    at = _fresh_app()
    assert len(at.success) + len(at.info) + len(at.warning) >= 6


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels
    for index in range(len(labels)):
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (labels[index], [str(e) for e in at.exception])


def test_slider_at_min_and_max():
    for edge in ("min", "max"):
        at = _fresh_app()
        s = at.sidebar.slider[0]
        s.set_value(s.min if edge == "min" else s.max).run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("key,optionen", [("wert_select", C.WERT_OPTIONEN), ("klasse_radio", C.KLASSEN), ("n_radio", C.N_OPTIONEN), ("dk_radio", C.DOSIS_KLASSEN), ("k_radio", C.K_OPTIONEN)])
def test_jede_auswahl_laeuft_ohne_fehler(key, optionen):
    at = _fresh_app()
    for o in optionen:
        regler = next(w for art in ("selectbox", "radio", "select_slider") for w in getattr(at.sidebar, art) if w.key == key)
        regler.set_value(o).run(timeout=TIMEOUT)
        assert not at.exception, (key, o, [str(e) for e in at.exception])


def test_anteil_regler_ueber_das_ganze_raster():
    at = _fresh_app()
    for p0 in C.P0_RASTER + (0.05, 0.85):
        at.sidebar.slider[0].set_value(p0).run(timeout=TIMEOUT)
        assert not at.exception, (p0, [str(e) for e in at.exception])
    assert "bei einem Anteil von 85 %" in _texte(at)
