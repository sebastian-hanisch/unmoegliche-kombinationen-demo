"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, verschiedene Zustände; Hilfetexte als Behauptungen (nachgerechnet); Permalink-Begrenzung."""

import pytest

import ukb_constants as C
import ukb_experiments as E
import ukb_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_and_default_preset_is_first_and_empty():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            assert (value in spec.optionen) if spec.optionen is not None else (spec.lo <= value <= spec.hi), (name, key)
    assert next(iter(C.PRESETS)).startswith("Standard") and C.PRESETS[next(iter(C.PRESETS))] == {}
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) and len({s.url_param for s in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_presets_unterscheiden_sich_im_zustand():
    zust = {tuple(sorted((k, str(p.get(k, P.SETTING_SPECS[sk].default))) for k, sk in P.PRESET_KEYS.items())) for p in C.PRESETS.values()}
    assert len(zust) == len(C.PRESETS)


def test_hilfetexte_als_behauptungen():
    w = E.wahrheiten(0.0)
    assert w["pd"] == pytest.approx(w["real"], abs=1e-12)                                              # „Anteil 0: die PD liegt auf dem realen Effekt“
    r = E.exp2("lineares Modell", 3000)
    assert max(r["sl_pd"]) - min(r["sl_pd"]) < 0.01 and r["sl_pd"] == pytest.approx(r["sl_ale"], abs=1e-9)      # „PD, ALE gleich“ beim linearen Modell
    assert E.exp3("Netz", 3000)["flach"][0] >= 5 and E.exp3("Netz", 3000)["flach"][1] <= 31          # „6 bis 30 % bei n = 3 000“ (Hilfetext)
    assert E.exp5("Boosting", 5)["hi0"][0] > 2.0                                                      # „mit fünf Fächern reicht das erste Fach in den ausgewerteten Bereich“


def test_permalink_begrenzt_zahlen_und_ignoriert_unbekannte_auswahl():
    q = P.SETTING_SPECS["p0_slider"]
    assert P._begrenzen(q, 5.0) == C.P0_MAX and P._begrenzen(q, -1.0) == C.P0_MIN
    assert P._begrenzen(P.SETTING_SPECS["n_radio"], 300) == 300 and P._begrenzen(P.SETTING_SPECS["n_radio"], 1000) is None
