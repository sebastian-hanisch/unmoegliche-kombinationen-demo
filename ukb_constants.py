"""Konstanten: Regler-Grenzen, Voreinstellungen, Farben und Presets."""

from fractions import Fraction

WERT_OPTIONEN = ("Vorlauf = 0", "Vorlauf = 0,5", "Vorlauf = 4", "Vorlauf = 12", "Art = Import", "Art = Export", "Art = Umschlag", "Art = Leer", "Zoll = 1")
DEFAULT_WERT = "Vorlauf = 4"
P0_MIN, P0_MAX, DEFAULT_P0, P0_STEP = 0.0, 0.9, 0.45, 0.05
KLASSEN = ("Boosting", "Random Forest", "Netz", "lineares Modell")
DEFAULT_KLASSE = "Boosting"
N_OPTIONEN = (300, 3000)
DEFAULT_N = 3000
DOSIS_KLASSEN = ("Boosting", "Netz")
DEFAULT_DK = "Boosting"
K_OPTIONEN = (5, 10, 20, 50)
DEFAULT_K = 20
P0_RASTER = (0.0, 0.2, 0.45, 0.7, 0.9)

COLOR_PD = "#C0504D"
COLOR_ALE = "#2F6B65"
COLOR_REAL = "#4c78a8"
COLOR_MODELL = "#8c6d31"
COLOR_GRAU = "rgba(150,150,150,0.55)"

PRESETS = {
    "Standard (45 % ohne Vorlauf, Boosting, n = 3000)": {},
    "Alle Zeilen haben einen Vorlauf (Anteil 0)": {"p0": 0.0},
    "Fast keine Zeilen mit Vorlauf (Anteil 0,9)": {"p0": 0.9},
    "Lineares Modell": {"klasse": "lineares Modell"},
    "Netz": {"klasse": "Netz", "dk": "Netz"},
    "Zu grobe ALE (K = 5)": {"k": 5},
}
PRESET_HELP = {
    "Standard (45 % ohne Vorlauf, Boosting, n = 3000)": "Der Generator mit 45 % Importen und Leercontainern (nie ein Vorlauf), Boosting, 3 000 Trainingszeilen.",
    "Alle Zeilen haben einen Vorlauf (Anteil 0)": "Kein Import und kein Leercontainer: die partielle Abhängigkeit liegt auf dem realen Effekt, und die ALE hat keinen Vorteil, der für beide Klassen gilt.",
    "Fast keine Zeilen mit Vorlauf (Anteil 0,9)": "Nur jede zehnte Zeile hat einen Vorlauf: die ALE ist beim Boosting in 4 von 5 Gruppen schlechter als die partielle Abhängigkeit, beim Netz in 4 von 5 besser.",
    "Lineares Modell": "Ein additives Modell kann nicht zwischen möglichen und unmöglichen Zeilen unterscheiden: PD, ALE und die PD über die Stütze sind gleich.",
    "Netz": "Das kleine Netz: ein erfundener Effekt auf unmöglichen Zeilen, aber ein Teil der unmöglichen Kurven ist flach (6 bis 30 % bei n = 3 000).",
    "Zu grobe ALE (K = 5)": "Angefordert sind fünf Fächer (wegen der Nullen bleiben drei): das erste Fach reicht bis in den ausgewerteten Bereich und die ALE-Kurve des Boostings weicht deutlich ab.",
}
