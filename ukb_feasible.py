"""Gültigkeit (Träger der Verteilung) und exakte Anteile unmöglicher Zeilen.

**Unmögliche Zeile** = Kombination, die der Generator nie erzeugt: Import oder Leer mit Vorlauf größer als 0, Export mit Vorlauf außerhalb [1, 10], Umschlag außerhalb [0,5; 6],
Zoll bei etwas anderem als Import. **Stütze** = Menge der möglichen Kombinationen.
`unmoeglich_anteil` rechnet den Anteil exakt (Brüche), wenn man **ein** Merkmal für **alle** Zeilen auf einen Wert setzt; die Tests prüfen ihn gegen eine Aufzählung der Zellen und gegen die Stichprobe."""

from fractions import Fraction

import numpy as np

import ukb_generator as G


def gueltig(X):
    """True, wenn die Kombination im Generator vorkommen kann."""
    art = X[:, 0].astype(int)
    v, zoll = X[:, 4], X[:, 2]
    ok = np.ones(len(X), dtype=bool)
    ok &= ~(np.isin(art, (0, 3)) & (v != 0))
    ok &= ~((art == 1) & ((v < 1) | (v > 10)))
    ok &= ~((art == 2) & ((v < 0.5) | (v > 6)))
    ok &= ~((zoll == 1) & (art != 0))
    return ok


def unmoeglich_je_art(merkmal: str, wert, p0=Fraction(9, 20)):
    """Je ursprünglicher Art (Import, Export, Umschlag, Leer): (Anteil aller Zeilen, der nach dem Setzen unmöglich ist, Anteil aller Zeilen dieser Art). Exakt bei Bruch `p0`.
    `merkmal` ('vorlauf', 'art', 'zoll') wird für **alle** Zeilen auf `wert` gesetzt."""
    anteile = G.art_wahrscheinlichkeiten(p0)
    one, zero = (Fraction(1), Fraction(0)) if isinstance(p0, Fraction) else (1.0, 0.0)
    if merkmal == "vorlauf":
        v = Fraction(wert) if isinstance(p0, Fraction) else float(wert)
        unm = (one if v != 0 else zero, zero if 1 <= v <= 10 else one, zero if Fraction(1, 2) <= v <= 6 else one, one if v != 0 else zero)
    elif merkmal == "zoll":
        unm = (zero, one, one, one) if int(wert) == 1 else (zero, zero, zero, zero)       # Zoll = 1 gibt es nur bei Import
    elif merkmal == "art":
        unm = {                                                             # Anteil je ursprünglicher Art, der nach dem Setzen auf Art `wert` unmöglich ist
            0: (zero, one, one, zero),                                      # Art = Import: Export und Umschlag (Vorlauf > 0) unmöglich
            1: (one, zero, one * Fraction(1, 11), one),                     # Art = Export: Import, Leer (Vorlauf 0) unmöglich; Umschlag bei Vorlauf < 1 (1/11 der Zeilen)
            2: (one, one * Fraction(4, 9), zero, one),                      # Art = Umschlag: Export bei Vorlauf > 6 (4/9 der Zeilen), Import, Leer unmöglich
            3: (one * Fraction(1, 4), one, one, zero),                      # Art = Leer: Import mit Zoll (1/4) unmöglich
        }[int(wert)]
    else:
        raise ValueError(merkmal)
    return [(anteile[g] * unm[g], anteile[g]) for g in range(4)]


def unmoeglich_anteil(merkmal: str, wert, p0=Fraction(9, 20)):
    """Exakter Anteil unmöglicher Zeilen, wenn `merkmal` für alle Zeilen auf `wert` gesetzt wird (Summe von `unmoeglich_je_art`)."""
    return sum((u for u, _ in unmoeglich_je_art(merkmal, wert, p0)), Fraction(0) if isinstance(p0, Fraction) else 0.0)
