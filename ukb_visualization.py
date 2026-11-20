"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import ukb_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=330, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend, legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def fig_karte(je_art, labels):
    """Gestapelte Säulen: je ursprünglicher Art der Anteil aller Zeilen, der nach dem Setzen möglich bzw. unmöglich ist."""
    unm = [float(u) for u, _ in je_art]
    mog = [float(a - u) for u, a in je_art]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=mog, name="möglich", marker_color=C.COLOR_ALE, hovertemplate="%{x}: %{y:.3f}<extra>möglich</extra>"))
    fig.add_trace(go.Bar(x=labels, y=unm, name="unmöglich (gab es nie)", marker_color=C.COLOR_PD, hovertemplate="%{x}: %{y:.3f}<extra>unmöglich</extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Anteil aller Container", range=[0, 0.7])
    fig.update_xaxes(title_text="ursprüngliche Art der Zeile")
    return _base(fig)


def fig_kurven(gitter_rel, wahr, modell_pd, modell_ale, klasse):
    """Anstieg des Vorlaufs relativ zu 2: Wahrheit je Funktional (PD, real, ALE) und das Modell (PD, ALE), Mittel über die fünf Gruppen."""
    xs = [2, 3, 4, 5, 6]
    fig = go.Figure()
    for nm, ys, farbe, dash in (("PD der Wahrheit", wahr["pd"], C.COLOR_PD, "dash"), ("realer Effekt", wahr["real"], C.COLOR_REAL, "dash"), ("ALE der Wahrheit", wahr["ale"], C.COLOR_ALE, "dash"),
                                (f"{klasse}: PD", modell_pd, C.COLOR_PD, "solid"), (f"{klasse}: ALE", modell_ale, C.COLOR_ALE, "solid")):
        fig.add_trace(go.Scatter(x=xs, y=[0.0] + list(ys), mode="lines+markers", name=nm, line=dict(color=farbe, dash=dash), hovertemplate="Vorlauf %{x}: %{y:.3f}<extra>" + nm + "</extra>"))
    fig.update_xaxes(title_text="Vorlauf (Tage)", dtick=1)
    fig.update_yaxes(title_text="Anstieg der ln-Verweilzeit gegen Vorlauf 2")
    return _base(fig, 380)


def fig_dosis(zeilen):
    """PD der Wahrheit gegen realen Effekt über den Anteil ohne Vorlauf."""
    xs = [z["p0"] for z in zeilen]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[z["pd_t"] for z in zeilen], mode="lines+markers", name="PD der Wahrheit", line=dict(color=C.COLOR_PD)))
    fig.add_trace(go.Scatter(x=xs, y=[z["real"] for z in zeilen], mode="lines+markers", name="realer Effekt", line=dict(color=C.COLOR_REAL)))
    fig.update_xaxes(title_text="Anteil der Container ohne Vorlauf")
    fig.update_yaxes(title_text="Anstieg Vorlauf 2 auf 5", rangemode="tozero")
    return _base(fig, 300)
