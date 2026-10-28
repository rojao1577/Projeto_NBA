import plotly.graph_objects as go
from analysis.descriptive import describe

VARIABLE_NAMES = {
    "pontos": "Pontos",
    "rebotes": "Rebotes",
    "assistencias": "Assistências",
    "roubos": "Roubos de bola",
    "tocos": "Tocos",
    "faltas": "Faltas",
    "cestas_3_convertidas": "Cestas de 3 convertidas",
}
TEAM_COLORS = {"Cleveland Cavaliers": "#2a78d6", "New York Knicks": "#eb6834"}
OTHER_COLOR = "#8a8984"
INK = "#3a3936"
BOX_WIDTH = 0.5


def _transparent(hex_color: str, alpha: float = 0.3) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _num(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def _first_time(fig: go.Figure, name: str) -> bool:
    return name not in {trace.name for trace in fig.data}


def _add_line(fig, x, y, name, dash, width):
    half = BOX_WIDTH / 2
    fig.add_trace(go.Scatter(
        x=[x - half, x + half], y=[y, y], mode="lines", name=name, legendgroup=name,
        showlegend=_first_time(fig, name), line=dict(color=INK, dash=dash, width=width),
        hovertext=f"{name}: {_num(y)}", hoverinfo="text",
    ))


def _add_points(fig, x, values, name, symbol, size, color, hover):
    fig.add_trace(go.Scatter(
        x=[x] * len(values), y=values, mode="markers", name=name, legendgroup=name,
        showlegend=_first_time(fig, name), marker=dict(symbol=symbol, size=size, color=color),
        hovertext=hover, hoverinfo="text",
    ))


def boxplot(df, teams, col):
    name = VARIABLE_NAMES[col]
    fig = go.Figure()
    for x, team in enumerate(teams):
        team_games = df[df["time"] == team]
        s = team_games[col]
        stats = describe(s)
        low, high = stats["limite_inferior"], stats["limite_superior"]
        inside = s[(s >= low) & (s <= high)]
        color = TEAM_COLORS.get(team, OTHER_COLOR)

        fig.add_trace(go.Box(
            x=[x], q1=[stats["q1"]], median=[stats["mediana"]], q3=[stats["q3"]],
            lowerfence=[inside.min()], upperfence=[inside.max()],
            width=BOX_WIDTH, whiskerwidth=0.6, fillcolor=_transparent(color),
            line=dict(color=color, width=2), name="Caixa: de Q1 a Q3",
            legendgroup="caixa", showlegend=x == 0, hoverinfo="skip",
        ))
        fig.add_annotation(x=x + BOX_WIDTH / 2, y=stats["q3"], text=f"Q3 = {_num(stats['q3'])}",
                           xanchor="left", showarrow=False, font=dict(size=11, color=INK))
        fig.add_annotation(x=x + BOX_WIDTH / 2, y=stats["q1"], text=f"Q1 = {_num(stats['q1'])}",
                           xanchor="left", showarrow=False, font=dict(size=11, color=INK))

        _add_line(fig, x, high, "Limite superior (Q3 + 1,5 × IQR)", "dash", 1.5)
        _add_line(fig, x, stats["mediana"], "Mediana", "solid", 3)
        _add_line(fig, x, low, "Limite inferior (Q1 − 1,5 × IQR)", "dot", 1.5)
        _add_points(fig, x, [stats["media"]], "Média", "diamond", 11, "#0b0b0b",
                    [f"Média: {_num(stats['media'])}"])
        _add_points(fig, x, stats["moda"], "Moda", "star", 14, "#eda100",
                    [f"Moda: {m}" for m in stats["moda"]])

        for label, games, outlier_color in [("Outliers superiores", team_games[s > high], "#e34948"),
                                            ("Outliers inferiores", team_games[s < low], "#6250d6")]:
            if len(games):
                hover = [f"{label}: {v}<br>{d:%d/%m/%Y} · {c}"
                         for d, c, v in zip(games["data_jogo"], games["confronto"], games[col])]
                _add_points(fig, x, games[col].tolist(), label, "circle-open", 11, outlier_color, hover)

    fig.update_layout(
        title=f"Boxplot — {name} por jogo",
        xaxis=dict(title="Time", tickvals=list(range(len(teams))), ticktext=teams,
                   range=[-0.6, len(teams) - 0.2]),
        yaxis_title=f"{name} (quantidade por jogo)",
        template="plotly_white", height=560, separators=",.",
        legend=dict(orientation="h", y=-0.18),
    )
    return fig
