import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analysis.descriptive import describe, league_zscores
from analysis.formatting import number

VARIABLE_NAMES = {
    "pontos": "Pontos",
    "rebotes": "Rebotes",
    "assistencias": "Assistências",
    "roubos": "Roubos de bola",
    "tocos": "Tocos",
    "faltas": "Faltas",
    "cestas_3_convertidas": "Cestas de 3 convertidas",
}
TEAM_COLORS = {"Cleveland Cavaliers": "#a3123a", "New York Knicks": "#eb6834"}
OTHER_COLOR = "#8a8984"
OTHER_COLORS = [OTHER_COLOR, "#1baf7a"]
INK = "#3a3936"
OUTLIER_COLOR = "#6250d6"
BOX_WIDTH = 0.5


def _transparent(hex_color: str, alpha: float = 0.3) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def team_color(team: str, position: int) -> str:
    return TEAM_COLORS.get(team, OTHER_COLORS[position % len(OTHER_COLORS)])


def _first_time(fig: go.Figure, name: str) -> bool:
    return name not in {trace.name for trace in fig.data}


def _add_line(fig, x, y, name, dash, width):
    half = BOX_WIDTH / 2
    fig.add_trace(go.Scatter(
        x=[x - half, x + half], y=[y, y], mode="lines", name=name, legendgroup=name,
        showlegend=_first_time(fig, name), line=dict(color=INK, dash=dash, width=width),
        hovertext=f"{name}: {number(y)}", hoverinfo="text",
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
        color = team_color(team, x)

        fig.add_trace(go.Box(
            x=[x], q1=[stats["q1"]], median=[stats["mediana"]], q3=[stats["q3"]],
            lowerfence=[inside.min()], upperfence=[inside.max()],
            width=BOX_WIDTH, whiskerwidth=0.6, fillcolor=_transparent(color),
            line=dict(color=color, width=2), name="Caixa: de Q1 a Q3",
            legendgroup="caixa", showlegend=x == 0, hoverinfo="skip",
        ))
        fig.add_annotation(x=x + BOX_WIDTH / 2, y=stats["q3"], text=f"Q3 = {number(stats['q3'])}",
                           xanchor="left", showarrow=False, font=dict(size=11, color=INK))
        fig.add_annotation(x=x + BOX_WIDTH / 2, y=stats["q1"], text=f"Q1 = {number(stats['q1'])}",
                           xanchor="left", showarrow=False, font=dict(size=11, color=INK))

        _add_line(fig, x, high, "Limite superior (Q3 + 1,5 × IQR)", "dash", 1.5)
        _add_line(fig, x, stats["mediana"], "Mediana", "solid", 3)
        _add_line(fig, x, low, "Limite inferior (Q1 − 1,5 × IQR)", "dot", 1.5)
        _add_points(fig, x, [stats["media"]], "Média", "diamond", 11, "#0b0b0b",
                    [f"Média: {number(stats['media'])}"])
        _add_points(fig, x, stats["moda"], "Moda", "star", 14, "#eda100",
                    [f"Moda: {m}" for m in stats["moda"]])

        for label, games, symbol in [("Outliers superiores", team_games[s > high], "triangle-up-open"),
                                     ("Outliers inferiores", team_games[s < low], "triangle-down-open")]:
            if len(games):
                hover = [f"{label}: {v}<br>{d:%d/%m/%Y} · {c}"
                         for d, c, v in zip(games["data_jogo"], games["confronto"], games[col])]
                _add_points(fig, x, games[col].tolist(), label, symbol, 12, OUTLIER_COLOR, hover)

    fig.update_layout(
        title=f"Boxplot — {name} por jogo",
        xaxis=dict(title="Time", tickvals=list(range(len(teams))), ticktext=teams,
                   range=[-0.6, len(teams) - 0.2]),
        yaxis_title=f"{name} (quantidade por jogo)",
        template="plotly_white", height=560, separators=",.",
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


def mean_bars(df, teams, columns):
    means = df.groupby("time")[columns].mean()
    league = means.mean()
    fig = make_subplots(rows=2, cols=4, subplot_titles=[VARIABLE_NAMES[col] for col in columns],
                        vertical_spacing=0.2, horizontal_spacing=0.07)
    for i, col in enumerate(columns):
        row, cell = i // 4 + 1, i % 4 + 1
        for position, team in enumerate(teams):
            value = means.loc[team, col]
            fig.add_trace(go.Bar(
                x=[team], y=[value], name=team, legendgroup=team, showlegend=i == 0,
                marker=dict(color=team_color(team, position), cornerradius=4),
                hovertext=f"{team}<br>{VARIABLE_NAMES[col]}: {number(value)} por jogo<br>"
                          f"Média da liga: {number(league[col])}",
                hoverinfo="text",
            ), row=row, col=cell)
        fig.add_hline(y=league[col], line=dict(color=INK, dash="dash", width=1.5), row=row, col=cell)
        fig.update_xaxes(showticklabels=False, row=row, col=cell)
        fig.update_yaxes(rangemode="tozero", row=row, col=cell)

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Média da liga (30 times)",
                             line=dict(color=INK, dash="dash", width=1.5)), row=1, col=1)
    fig.update_yaxes(title_text="Média por jogo", row=1, col=1)
    fig.update_yaxes(title_text="Média por jogo", row=2, col=1)
    fig.update_layout(
        title="Médias por jogo — cada variável na sua própria escala",
        template="plotly_white", height=620, separators=",.", bargap=0.35,
        legend=dict(orientation="h", y=-0.08),
    )
    return fig


def zscore_bars(df, teams, columns):
    z = league_zscores(df, teams, columns)
    means = df.groupby("time")[columns].mean()
    ranks = means.rank(ascending=False).astype(int)
    names = [VARIABLE_NAMES[col] for col in columns]
    fig = go.Figure()
    for position, team in enumerate(teams):
        hover = [f"{team}<br>{VARIABLE_NAMES[col]}: z = {number(z.loc[team, col])}<br>"
                 f"Média do time: {number(means.loc[team, col])} · liga: {number(means[col].mean())}<br>"
                 f"Posição na liga: {ranks.loc[team, col]}º de {len(means)} (1º = maior valor)"
                 for col in columns]
        fig.add_trace(go.Bar(
            y=names, x=z.loc[team].tolist(), orientation="h", name=team,
            marker=dict(color=team_color(team, position), cornerradius=4),
            text=[number(v) for v in z.loc[team]], textposition="outside",
            hovertext=hover, hoverinfo="text",
        ))
    limit = max(1.5, float(z.abs().max().max()) + 0.6)
    fig.add_vline(x=0, line=dict(color=INK, width=1.5),
                  annotation_text="time típico da liga (z = 0)", annotation_position="top")
    fig.update_layout(
        title="Comparação padronizada — z-score da média do time em relação à liga",
        xaxis=dict(title="z-score (desvios padrão em relação à média dos 30 times)",
                   range=[-limit, limit], zeroline=False),
        yaxis=dict(autorange="reversed"),
        barmode="group", template="plotly_white", height=560, separators=",.",
        legend=dict(orientation="h", y=-0.18),
    )
    return fig
