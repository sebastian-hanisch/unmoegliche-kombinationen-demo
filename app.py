"""Unmögliche Kombinationen: was die Kurve fragt, wenn es die Zeile nie gab - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Partielle Abhängigkeit (PD), ICE und ALE des Vorlaufs in einem erfundenen Container-Beispiel, in dem bei 45 % der Container (Import und Leer) nie ein Vorlauf vorkommt. Exakte Anteile und die Wahrheitskurven
werden live gerechnet; Messungen von vier Modellklassen sind eingefroren (`data/`). Es gibt keine reale Datenquelle.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import ukb_constants as C
import ukb_evaluation as ev
import ukb_experiments as E
import ukb_texts as TX
import ukb_visualization as V
from ukb_layout import dunkles_design, kopf_css
from ukb_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Unmögliche Kombinationen: was die Kurve fragt, wenn es die Zeile nie gab – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _exp1(label, p0):
    return E.exp1(label, p0)


@st.cache_data(show_spinner=False)
def _wahr(p0=0.45):
    return E.wahrheiten(p0)


@st.cache_data(show_spinner=False)
def _wahr_kurven():
    import ukb_curves as CV
    return {"pd": CV.anstieg(CV.pd_wahrheit()).tolist(), "real": CV.anstieg(CV.real_wahrheit()).tolist(), "ale": CV.anstieg(CV.ale_wahrheit()).tolist()}


@st.cache_data(show_spinner=False)
def _exp2(klasse, n):
    return E.exp2(klasse, n)


@st.cache_data(show_spinner=False)
def _exp3(klasse, n):
    return E.exp3(klasse, n)


@st.cache_data(show_spinner=False)
def _linear(n):
    return E.linear_live(n)


@st.cache_data(show_spinner=False)
def _exp4(dk, p0):
    return E.exp4(dk, p0)


@st.cache_data(show_spinner=False)
def _exp5(dk, k):
    return E.exp5(dk, k)


def _md_tabelle(kopf, zeilen):
    out = ["| " + " | ".join(kopf) + " |", "|" + "|".join("---" for _ in kopf) + "|"]
    out += ["| " + " | ".join(str(c) for c in z) + " |" for z in zeilen]
    return "\n".join(out)


st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 " + TX.TITEL)
st.markdown(TX.EINLEITUNG)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/unmoegliche-kombinationen-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Eine Kurve zur Frage „Wie wirkt der Vorlauf?“ entsteht so: Man setzt den Vorlauf für **alle** Container auf einen Wert und mittelt die Antworten des Modells. Dabei kommen auch Container vor, die es nie gab, etwa ein Importcontainer mit Vorlauf. Dort hat das Modell nie ein Beispiel gesehen und liefert eine ausgedachte Antwort.

Die Demo zeigt, wie viele solche Fälle es in diesem erfundenen Containerstrom gibt, wie vier Modellarten dort danebenliegen und wie eine Variante (die **ALE**) näher bei den Daten bleibt, aber nicht ganz. Die Container sind erfunden, die Aussagen gelten für diesen Generator, nicht für echte Container.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/unmoegliche-kombinationen-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert die Frage nach der Kurve", expanded=True):
    st.markdown(TX.SO_FUNKTIONIERT)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🗺️ Experiment 1: Karte")
    wert = st.selectbox("Merkmal für alle Zeilen setzen", C.WERT_OPTIONEN, key="wert_select", help="Welche Kombination fragt die Kurve? Eine Auswahl wird für alle Container gesetzt.")
    p0 = st.slider("Anteil der Container ohne Vorlauf (Import und Leer)", *bounds("p0_slider"), step=C.P0_STEP, key="p0_slider", format="%.2f",
                   help="Der Generator mit diesem Anteil; Standard 0,45. Wirkt auf Experiment 1 und 4 (Experimente 2, 3 und 5 sind bei 0,45 eingefroren).")
    st.header("📈 Experimente 2 und 3: Kurven und ICE")
    klasse = st.radio("Modellklasse", C.KLASSEN, key="klasse_radio", help="Boosting, Random Forest, Netz (16 Einheiten) und lineares Modell, eingefroren.")
    n = st.radio("Trainingsgröße n", C.N_OPTIONEN, key="n_radio", horizontal=True, help="Zahl der Trainingszeilen je Zufalls-Seed.")
    st.header("🧪 Experimente 4 und 5: Dosis und Intervallzahl")
    dk = st.radio("Klasse", C.DOSIS_KLASSEN, key="dk_radio", horizontal=True, help="Boosting oder Netz (n = 1 000).")
    k = st.radio("Zahl der Intervalle K der ALE", C.K_OPTIONEN, key="k_radio", horizontal=True, help="Angeforderte Zahl der Quantilintervalle des Vorlaufs; bei 45 % Nullen bleiben weniger (bei K = 20 elf bis zwölf Intervalle).")

sync_query_params({"wert_select": wert, "p0_slider": float(p0), "klasse_radio": klasse, "n_radio": int(n), "dk_radio": dk, "k_radio": int(k)})

# ----------------------------------------------------------------------------------------------- Experiment 1
st.markdown("## ❓ Kernfrage 1: Wie viele Zeilen gibt es nicht?")
st.subheader("🔬 Experiment 1: Die Karte der Kombinationen")
st.caption(TX.EXP1_CAPTION)
r1 = _exp1(wert, float(p0))
c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(V.fig_karte(r1["je_art"], ["Import", "Export", "Umschlag", "Leer"]), width="stretch", key="chart_exp1")
with c2:
    a, b = st.columns(2)
    a.metric("exakt (Bruch)", ev.bruch(r1["exakt"]))
    b.metric("exakt (Prozent)", ev.de(100 * float(r1["exakt"]), 1) + " %")
    st.metric("gezählt (6 000 Zeilen)", ev.de(100 * r1["stichprobe"], 1) + " %", delta=f"Standardfehler {ev.de(100 * r1['se'], 1)} Prozentpunkte", delta_color="off")
kind, text = ev.verdict_karte(wert, r1)
getattr(st, kind)(text)

st.markdown("---")

# ----------------------------------------------------------------------------------------------- Experiment 2
st.markdown("## ❓ Kernfrage 2: Welche Kurve zeigt den Effekt des Vorlaufs?")
st.subheader("🔬 Experiment 2: PD, real und ALE, eine Kurve, drei Antworten")
st.caption(TX.EXP2_CAPTION)
r2 = _exp2(klasse, int(n))
wk = _wahr_kurven()
st.plotly_chart(V.fig_kurven(None, wk, r2["pd_inc"], r2["ale_inc"], klasse), width="stretch", key="chart_exp2")
w0 = _wahr()
a, b, c, d_ = st.columns(4)
a.metric("PD des Modells, 2 auf 5", ev.spanne(*r2["sl_pd"], 2))
b.metric("ALE des Modells, 2 auf 5", ev.spanne(*r2["sl_ale"], 2))
c.metric("PD / real / ALE der Wahrheit", f"{ev.de(w0['pd'], 2)} / {ev.de(w0['real'], 2)} / {ev.de(w0['ale'], 2)}")
d_.metric("unmögliche Auswertungspunkte der ALE", ev.spanne(100 * r2["inv_ale"][0], 100 * r2["inv_ale"][1], 1) + " %")
kind, text = ev.verdict_kurven(r2, w0)
getattr(st, kind)(text)
st.caption(TX.NEGATIV_2)

st.markdown("---")

# ----------------------------------------------------------------------------------------------- Experiment 3
st.markdown("## ❓ Kernfrage 3: Was zeigt die ICE auf Zeilen, die es nie gab?")
st.subheader("🔬 Experiment 3: ICE, der erfundene Effekt")
st.caption(TX.EXP3_CAPTION)
r3 = _exp3(klasse, int(n))
a, b, c = st.columns(3)
a.metric("mittlerer Anstieg 2 auf 5 auf unmöglichen Zeilen (Wahrheit 0)", ev.spanne(*r3["ice_imp"], 2))
b.metric("flache Kurven (Anstieg unter 0,1)", ev.spanne(*r3["flach"], 1) + " %")
c.metric("Fehler bei Vorlauf 5: unmöglich gegen möglich", ev.spanne(*r3["fehler_verh"], 1) + "-mal")
kind, text = ev.verdict_ice(r3)
getattr(st, kind)(text)
kind, text = ev.verdict_linear_live(_linear(int(n)))
getattr(st, kind)(text)
st.caption(TX.NEGATIV_3)

st.markdown("---")

# ----------------------------------------------------------------------------------------------- Experiment 4
st.markdown("## ❓ Kernfrage 4: Wie viel hängt am Anteil?")
st.subheader("🔬 Experiment 4: Dosis, der Anteil ohne Vorlauf")
st.caption(TX.EXP4_CAPTION)
r4 = _exp4(dk, float(p0))
st.plotly_chart(V.fig_dosis(r4["zeilen"]), width="stretch", key="chart_exp4")
st.markdown(_md_tabelle(
    ["Anteil", "PD der Wahrheit (Stichprobe)", "realer Effekt", "Abweichung PD gegen real (Spanne)", "Abweichung ALE gegen real (Spanne)", "ALE besser in Gruppen"],
    [[ev.de(z["p0"], 2), ev.de(z["pd_t"], 3), ev.de(z["real"], 3), ev.spanne(*z["eP"], 2), ev.spanne(*z["eA"], 2), f"{z['ale_besser_gruppen']} von 5"] for z in r4["zeilen"]]))
kind, text = ev.verdict_dosis(r4)
getattr(st, kind)(text)
st.caption(TX.NEGATIV_4)

st.markdown("---")

# ----------------------------------------------------------------------------------------------- Experiment 5
st.markdown("## ❓ Kernfrage 5: Fragt die ALE nie nach Unmöglichem?")
st.subheader("🔬 Experiment 5: ALE ist nicht gratis")
st.caption(TX.EXP5_CAPTION)
r5 = _exp5(dk, int(k))
a, b, c = st.columns(3)
a.metric("unmögliche Auswertungspunkte", ev.spanne(100 * r5["inv"][0], 100 * r5["inv"][1], 1) + " %")
b.metric("Effekt im ersten Intervall (Modell)", ev.spanne(*r5["eff0_m"], 2))
c.metric("erstes Intervall reicht bis Vorlauf", ev.spanne(*r5["hi0"], 2))
kind, text = ev.verdict_faecher(r5)
getattr(st, kind)(text)
st.caption(TX.NEGATIV_5)

st.markdown("---")

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(TX.GRENZEN_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/unmoegliche-kombinationen-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind ein Modell $f$ (oder die wahre Funktion $\mu$) und Zeilen $x^{(1)}, \dots, x^{(n)}$ mit einem Merkmal $v$ (Vorlauf). Gesucht ist eine Kurve, die zeigt, wie die Vorhersage von $v$ abhängt.

| Symbol | Bedeutung |
|---|---|
| $f$ | Modell (Vorhersage der ln-Verweilzeit) |
| $\mu$ | wahre Mitte der ln-Verweilzeit (die bekannte Funktion des Generators) |
| $x_{\setminus v}^{(i)}$ | Zeile $i$ ohne den Vorlauf |
| $v$, $z$, $z_0$ | Vorlauf, Gitterwert, erster Gitterwert (Anfang der Kurve) |
| $\mathrm{real}(z)$ | realer Effekt: Mittel von $\mu$ über die Zeilen, in denen der Vorlauf vorkommt (Export und Umschlag), bei Vorlauf $z$ |
| $p_0$ | Anteil der Container ohne Vorlauf (Import und Leer) |
| $K$ | Zahl der Intervalle der ALE |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/unmoegliche-kombinationen-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    for text_, formel in TX.MATHE_BLOECKE:
        st.markdown(text_)
        st.latex(formel)
    st.markdown(TX.MATHE_SCHLUSS)
    st.markdown(TX.LITERATUR)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
