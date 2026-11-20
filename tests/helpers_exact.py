"""Exakte Orakel für die Tests (Brüche): sie rechnen **nicht** mit den Modulen der Demo.

* `anteil_enum`: Anteil unmöglicher Zeilen durch Aufzählung der Zellen des Generators (Art, Zoll, Halbtag-Zelle des Vorlaufs) mit der Gültigkeitsregel als Intervallenthaltung.
* `pd_exakt`, `ice_exakt`, `ale_exakt`: PD, ICE und ALE auf gewichteten Zeilen in Brüchen (Definitionen von Hand, ALE mit vorgegebenen Fächerkanten, das erste Fach enthält die kleinsten Werte).
Zeilen sind Tupel (Art, Vorlauf) mit Gewicht; ein Modell ist eine Funktion f(art, vorlauf)."""

from fractions import Fraction as Fr


def zellen(p0):
    """(Art, Zoll, Vorlauf-Zelle (untere Kante, Breite) oder None, Gewicht) des Generators bei Anteil p0 ohne Vorlauf; Brüche."""
    pi, pe, pu, pl = Fr(7, 9) * p0, Fr(6, 11) * (1 - p0), Fr(5, 11) * (1 - p0), Fr(2, 9) * p0
    out = [(0, 0, None, pi * Fr(3, 4)), (0, 1, None, pi * Fr(1, 4)), (3, 0, None, pl)]
    out += [(1, 0, (Fr(1) + Fr(k, 2), Fr(1, 2)), pe / 18) for k in range(18)]
    out += [(2, 0, (Fr(1, 2) + Fr(k, 2), Fr(1, 2)), pu / 11) for k in range(11)]
    assert sum(w for *_, w in out) == 1
    return out


def gueltig_zelle(art, zoll, zelle):
    if zoll == 1 and art != 0:
        return False
    if art in (0, 3):
        return zelle is None
    if zelle is None:
        return False
    lo, w = zelle
    a, b = (Fr(1), Fr(10)) if art == 1 else (Fr(1, 2), Fr(6))
    return a <= lo and lo + w <= b


def zelle_von(v):
    """Zelle (untere Kante, Breite 1/2) eines Vorlaufs v (Halbtag-Raster); 0 ist die Atommasse None."""
    v = Fr(v)
    if v == 0:
        return None
    return (Fr(int(v * 2), 2) if v * 2 == int(v * 2) else Fr(int(v * 2), 2), Fr(1, 2))


def anteil_enum(merkmal, wert, p0):
    tot = Fr(0)
    for art, zoll, zelle, w in zellen(p0):
        a, z, c = art, zoll, zelle
        if merkmal == "vorlauf":
            c = zelle_von(wert)
        elif merkmal == "art":
            a = int(wert)
        elif merkmal == "zoll":
            z = int(wert)
        if not gueltig_zelle(a, z, c):
            tot += w
    return tot


def pd_exakt(f, zeilen, gitter):
    """PD an den Gitterwerten: gewichtetes Mittel von f(art, g) über alle Zeilen (Art, Vorlauf, Gewicht)."""
    return [sum(w * f(a, g) for a, v, w in zeilen) for g in gitter]


def ice_exakt(f, zeile, von, bis):
    return f(zeile[0], bis) - f(zeile[0], von)


def ale_exakt(f, zeilen, kanten):
    """ALE-Effekte je Fach (gewichtetes Mittel der Differenzen f(obere Kante) − f(untere Kante) der Zeilen im Fach); Fach i = (kanten[i], kanten[i+1]], das erste schließt die kleinste Kante ein."""
    eff = []
    for i in range(len(kanten) - 1):
        lo, hi = kanten[i], kanten[i + 1]
        im_fach = [(a, v, w) for a, v, w in zeilen if (lo < v <= hi) or (i == 0 and v == lo)]
        wt = sum(w for _, _, w in im_fach)
        eff.append(sum(w * (f(a, hi) - f(a, lo)) for a, v, w in im_fach) / wt)
    return eff
