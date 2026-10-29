import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from analysis.descriptive import describe, league_zscores
from analysis.formatting import number
from analysis.shape import by_team, cv_table, poisson_cv_table

VARIABLE_NAMES = {
    "pontos": "Pontos",
    "rebotes": "Rebotes",
    "assistencias": "Assistências",
    "roubos": "Roubos de bola",
    "tocos": "Tocos",
    "faltas": "Faltas",
    "cestas_3_convertidas": "Cestas de 3 convertidas",
}
# cores oficiais (principal, secundária) — fonte: teamcolorcodes.com
TEAM_COLORS = {
    "Atlanta Hawks": ("#E03A3E", "#C1D32F"),
    "Boston Celtics": ("#007A33", "#BA9653"),
    "Brooklyn Nets": ("#000000", "#FFFFFF"),
    "Charlotte Hornets": ("#1D1160", "#00788C"),
    "Chicago Bulls": ("#CE1141", "#000000"),
    "Cleveland Cavaliers": ("#860038", "#041E42"),
    "Dallas Mavericks": ("#00538C", "#002B5E"),
    "Denver Nuggets": ("#0E2240", "#FEC524"),
    "Detroit Pistons": ("#C8102E", "#1D42BA"),
    "Golden State Warriors": ("#1D428A", "#FFC72C"),
    "Houston Rockets": ("#CE1141", "#000000"),
    "Indiana Pacers": ("#002D62", "#FDBB30"),
    "Los Angeles Clippers": ("#C8102E", "#1D428A"),
    "Los Angeles Lakers": ("#552583", "#FDB927"),
    "Memphis Grizzlies": ("#5D76A9", "#12173F"),
    "Miami Heat": ("#98002E", "#F9A01B"),
    "Milwaukee Bucks": ("#00471B", "#EEE1C6"),
    "Minnesota Timberwolves": ("#0C2340", "#236192"),
    "New Orleans Pelicans": ("#0C2340", "#C8102E"),
    "New York Knicks": ("#006BB6", "#F58426"),
    "Oklahoma City Thunder": ("#007AC1", "#EF3B24"),
    "Orlando Magic": ("#0077C0", "#C4CED4"),
    "Philadelphia 76ers": ("#006BB6", "#ED174C"),
    "Phoenix Suns": ("#1D1160", "#E56020"),
    "Portland Trail Blazers": ("#E03A3E", "#000000"),
    "Sacramento Kings": ("#5A2D81", "#63727A"),
    "San Antonio Spurs": ("#C4CED4", "#000000"),
    "Toronto Raptors": ("#CE1141", "#000000"),
    "Utah Jazz": ("#002B5C", "#00471B"),
    "Washington Wizards": ("#002B5C", "#E31837"),
}
MIN_COLOR_DISTANCE = 15
INK = "#3a3936"
OUTLIER_COLOR = "#6250d6"
BOX_WIDTH = 0.5


def _transparent(hex_color: str, alpha: float = 0.3) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _oklab(hex_color: str) -> tuple[float, float, float]:
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def color_distance(a: str, b: str) -> float:
    return 100 * float(np.linalg.norm(np.subtract(_oklab(a), _oklab(b))))


def team_palette(teams: list[str]) -> dict[str, tuple[str, str]]:
    palette = {}
    for team in teams:
        main, stripe = TEAM_COLORS[team]
        if any(color_distance(main, used) < MIN_COLOR_DISTANCE for used, _ in palette.values()):
            main, stripe = stripe, main
        palette[team] = (main, stripe)
    return palette


def team_marker(colors: tuple[str, str]) -> dict:
    main, stripe = colors
    return dict(color=main, cornerradius=4, line=dict(color=stripe, width=1),
                pattern=dict(shape="/", fgcolor=stripe, size=10, solidity=0.3, fillmode="overlay"))


def darker(colors: tuple[str, str]) -> str:
    return min(colors, key=lambda color: _oklab(color)[0])


def team_dot(colors: tuple[str, str]) -> str:
    main, stripe = colors
    return f"linear-gradient(135deg, {main} 50%, {stripe} 50%)"


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
    palette = team_palette(teams)
    for x, team in enumerate(teams):
        team_games = df[df["time"] == team]
        s = team_games[col]
        stats = describe(s)
        low, high = stats["limite_inferior"], stats["limite_superior"]
        inside = s[(s >= low) & (s <= high)]
        main, edge = palette[team][0], darker(palette[team])

        fig.add_trace(go.Box(
            x=[x], q1=[stats["q1"]], median=[stats["mediana"]], q3=[stats["q3"]],
            lowerfence=[inside.min()], upperfence=[inside.max()],
            width=BOX_WIDTH, whiskerwidth=0.6, fillcolor=_transparent(main, 0.5),
            line=dict(color=edge, width=2), name="Caixa: de Q1 a Q3",
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
    palette = team_palette(teams)
    fig = make_subplots(rows=2, cols=4, subplot_titles=[VARIABLE_NAMES[col] for col in columns],
                        vertical_spacing=0.2, horizontal_spacing=0.07)
    for i, col in enumerate(columns):
        row, cell = i // 4 + 1, i % 4 + 1
        for position, team in enumerate(teams):
            value = means.loc[team, col]
            fig.add_trace(go.Bar(
                x=[team], y=[value], name=team, legendgroup=team, showlegend=i == 0,
                marker=team_marker(palette[team]),
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
    palette = team_palette(teams)
    names = [VARIABLE_NAMES[col] for col in columns]
    fig = go.Figure()
    for position, team in enumerate(teams):
        hover = [f"{team}<br>{VARIABLE_NAMES[col]}: z = {number(z.loc[team, col])}<br>"
                 f"Média do time: {number(means.loc[team, col])} · liga: {number(means[col].mean())}<br>"
                 f"Posição na liga: {ranks.loc[team, col]}º de {len(means)} (1º = maior valor)"
                 for col in columns]
        fig.add_trace(go.Bar(
            y=names, x=z.loc[team].tolist(), orientation="h", name=team,
            marker=team_marker(palette[team]),
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


SEQUENTIAL_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def method_heatmap(counts, scope):
    names = [VARIABLE_NAMES[col] for col in counts.index]
    methods = counts.columns.tolist()
    hover = [[f"{name} · {method}: {counts.loc[col, method]} jogos marcados"
              for method in methods] for col, name in zip(counts.index, names)]
    fig = go.Figure(go.Heatmap(
        z=counts.values, x=methods, y=names, colorscale=SEQUENTIAL_BLUES, zmin=0,
        text=counts.values, texttemplate="%{text}", hovertext=hover, hoverinfo="text",
        xgap=2, ygap=2, colorbar=dict(title="Jogos<br>marcados", tickformat="d",
                                      dtick=1 if counts.values.max() <= 5 else None),
    ))
    fig.update_layout(
        title=f"Outliers por coluna e por método — {scope}",
        xaxis=dict(title="Método de detecção", side="bottom"),
        yaxis=dict(title="Variável (cada uma analisada sozinha)", autorange="reversed"),
        template="plotly_white", height=480, separators=",.",
    )
    return fig


def votes_bars(votes, scope):
    counts = votes.value_counts().reindex(range(1, 5), fill_value=0)
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values, marker=dict(color=OUTLIER_COLOR, cornerradius=4),
        text=counts.values, textposition="outside",
        hovertext=[f"{n} jogos marcados por {v} dos 4 métodos" for v, n in counts.items()],
        hoverinfo="text",
    ))
    fig.update_layout(
        title=f"Em quantos métodos multivariados cada jogo foi outlier — {scope}<br>"
              f"<sup>{int((votes == 0).sum())} jogos não foram marcados por nenhum método (não aparecem no gráfico)</sup>",
        xaxis=dict(title="Número de métodos que marcaram o jogo (ECOD, K-NN, LOF, DBSCAN)", tickvals=[1, 2, 3, 4]),
        yaxis=dict(title="Quantidade de jogos", rangemode="tozero"),
        template="plotly_white", height=460, separators=",.", bargap=0.4,
    )
    return fig


def method_strip(df, flags, col, rows, scope):
    name = VARIABLE_NAMES[col]
    games = df[rows]
    fig = go.Figure(go.Box(
        x=games[col], y=["Todos os jogos"] * len(games), orientation="h", name="Todos os jogos",
        marker_color="#8a8984", boxpoints=False, hoverinfo="skip",
    ))
    legend_shown = False
    for method in reversed(flags.columns):
        marked = games[flags.loc[rows, method]]
        show_legend = len(marked) > 0 and not legend_shown
        legend_shown |= show_legend
        fig.add_trace(go.Scatter(
            x=marked[col], y=[method] * len(marked), mode="markers", name="Jogo marcado",
            legendgroup="marcado", showlegend=show_legend,
            marker=dict(color=OUTLIER_COLOR, size=10, symbol="circle-open", line=dict(width=2)),
            hovertext=[f"{method}: {v}<br>{t}<br>{d:%d/%m/%Y} · {c} · {m} min"
                       for v, t, d, c, m in zip(marked[col], marked["time"], marked["data_jogo"],
                                                marked["confronto"], marked["minutos"])],
            hoverinfo="text",
        ))
    fig.update_layout(
        title=f"Quais valores de {name} cada método marcou — {scope}",
        xaxis_title=f"{name} (quantidade por jogo)",
        yaxis=dict(title="Método", categoryorder="array",
                   categoryarray=list(reversed(flags.columns)) + ["Todos os jogos"]),
        template="plotly_white", height=520, separators=",.",
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


def histogram_kde(df, teams, col):
    name = VARIABLE_NAMES[col]
    fig = make_subplots(rows=1, cols=len(teams), subplot_titles=teams, shared_yaxes=True)
    palette = team_palette(teams)
    for i, team in enumerate(teams):
        s = df[df["time"] == team][col]
        q1, q3 = s.quantile([0.25, 0.75])
        width = max(1, round(2 * (q3 - q1) / len(s) ** (1 / 3)))
        fig.add_trace(go.Histogram(
            x=s, histnorm="probability density", name="Histograma", legendgroup="hist", showlegend=i == 0,
            xbins=dict(start=s.min() - 0.5, end=s.max() + 0.5, size=width),
            marker=team_marker(palette[team]), opacity=0.8,
            hovertemplate=f"{team}<br>{name}: %{{x}}<br>Densidade: %{{y:.3f}}<extra></extra>",
        ), row=1, col=i + 1)
        grid = np.linspace(s.min() - width, s.max() + width, 200)
        fig.add_trace(go.Scatter(
            x=grid, y=gaussian_kde(s)(grid), mode="lines", name="Curva de densidade (KDE)",
            legendgroup="kde", showlegend=i == 0, line=dict(color=INK, width=2), hoverinfo="skip",
        ), row=1, col=i + 1)
        fig.add_vline(x=s.mean(), line=dict(color=INK, dash="dash", width=1.5), row=1, col=i + 1)
        fig.add_vline(x=s.median(), line=dict(color=INK, dash="dot", width=1.5), row=1, col=i + 1)
        fig.update_xaxes(title_text=f"{name} (quantidade por jogo)", row=1, col=i + 1)

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Média",
                             line=dict(color=INK, dash="dash", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Mediana",
                             line=dict(color=INK, dash="dot", width=1.5)), row=1, col=1)
    fig.update_yaxes(title_text="Densidade (proporção de jogos ÷ largura da faixa)", row=1, col=1)
    fig.update_layout(
        title=f"Histograma e curva de densidade — {name} por jogo",
        template="plotly_white", height=480, separators=",.", bargap=0.05,
        legend=dict(orientation="h", y=-0.22),
    )
    return fig


def cv_bars(df, teams, columns):
    names = [VARIABLE_NAMES[col] for col in columns]
    std = by_team(df, teams, columns, "std")
    cv = cv_table(df, teams, columns)
    expected = poisson_cv_table(df, teams, columns)
    palette = team_palette(teams)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=["Desvio padrão (na unidade de cada variável)",
                                        "Coeficiente de variação (%)"])
    for position, team in enumerate(teams):
        fig.add_trace(go.Bar(
            y=names, x=std[team], orientation="h", name=team, legendgroup=team, offsetgroup=str(position),
            marker=team_marker(palette[team]),
            hovertext=[f"{team}<br>{n}: desvio {number(v)}" for n, v in zip(names, std[team])],
            hoverinfo="text",
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            y=names, x=cv[team], orientation="h", name=team, legendgroup=team, showlegend=False,
            offsetgroup=str(position), marker=team_marker(palette[team]),
            hovertext=[f"{team}<br>{n}: CV {number(v)}%<br>Esperado numa contagem de Poisson: {number(e)}%"
                       for n, v, e in zip(names, cv[team], expected[team])],
            hoverinfo="text",
        ), row=1, col=2)

    for position, team in enumerate(teams):
        fig.add_trace(go.Scatter(
            y=names, x=expected[team], mode="markers", orientation="h", name="CV esperado só pelo tamanho da média (1 ÷ √média)",
            legendgroup="poisson", showlegend=position == 0, offsetgroup=str(position),
            marker=dict(symbol="diamond", size=10, color="#ffffff", line=dict(color=INK, width=2)),
            hovertext=[f"{team}<br>{n}: CV esperado {number(e)}%" for n, e in zip(names, expected[team])],
            hoverinfo="text",
        ), row=1, col=2)

    fig.update_xaxes(title_text="Desvio padrão amostral (n − 1)", rangemode="tozero", row=1, col=1)
    fig.update_xaxes(title_text="CV = desvio ÷ média × 100 (%)", rangemode="tozero",
                     range=[0, float(cv.max().max()) * 1.1], row=1, col=2)
    fig.update_layout(
        title="Dispersão bruta × dispersão relativa",
        yaxis=dict(autorange="reversed"), barmode="group", scattermode="group",
        template="plotly_white", height=560, separators=",.",
        legend=dict(orientation="h", y=-0.16),
    )
    return fig


def iqr_ranges(df, teams, columns):
    palette = team_palette(teams)
    fig = make_subplots(rows=2, cols=4, subplot_titles=[VARIABLE_NAMES[col] for col in columns],
                        vertical_spacing=0.2, horizontal_spacing=0.06)
    for i, col in enumerate(columns):
        row, cell = i // 4 + 1, i % 4 + 1
        for team in teams:
            s = df[df["time"] == team][col]
            q1, q3 = s.quantile([0.25, 0.75])
            fig.add_trace(go.Scatter(
                x=[s.min(), s.max()], y=[team, team], mode="lines+markers", name="Amplitude (mínimo a máximo)",
                legendgroup="amplitude", showlegend=i == 0 and team == teams[0],
                line=dict(color=INK, width=1.5), marker=dict(symbol="line-ns", size=10, line=dict(color=INK, width=1.5)),
                hovertext=f"{team}<br>Mínimo {number(s.min())} · máximo {number(s.max())}<br>"
                          f"Amplitude {number(s.max() - s.min())}",
                hoverinfo="text",
            ), row=row, col=cell)
            fig.add_trace(go.Bar(
                y=[team], x=[q3 - q1], base=[q1], orientation="h", name=team, legendgroup=team,
                showlegend=i == 0, marker=team_marker(palette[team]), width=0.5,
                hovertext=f"{team}<br>Q1 {number(q1)} · Q3 {number(q3)}<br>IQR {number(q3 - q1)}",
                hoverinfo="text",
            ), row=row, col=cell)
            fig.add_trace(go.Scatter(
                x=[s.median()], y=[team], mode="markers", name="Mediana", legendgroup="mediana",
                showlegend=i == 0 and team == teams[0],
                marker=dict(symbol="diamond", size=9, color="#ffffff", line=dict(color=INK, width=2)),
                hovertext=f"{team}<br>Mediana {number(s.median())}", hoverinfo="text",
            ), row=row, col=cell)
        fig.update_yaxes(showticklabels=False, autorange="reversed", row=row, col=cell)
        fig.update_xaxes(title_text="por jogo", title_font=dict(size=11), row=row, col=cell)

    fig.update_layout(
        title="Faixa central (IQR, de Q1 a Q3) dentro da amplitude (mínimo a máximo)",
        template="plotly_white", height=560, separators=",.", barmode="overlay",
        legend=dict(orientation="h", y=-0.14),
    )
    return fig


MONTH_ABBREVIATIONS = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
SINGLE_COLOR = "#2a78d6"


def date_bars(by_month, by_weekday):
    months = [f"{MONTH_ABBREVIATIONS[p.month - 1]}/{p.year}" for p in by_month.index]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.08,
                        subplot_titles=["Jogos por mês", "Jogos por dia da semana"])
    for x, counts, col in [(months, by_month, 1), (list(by_weekday.index), by_weekday, 2)]:
        fig.add_trace(go.Bar(
            x=x, y=counts.values, marker=dict(color=SINGLE_COLOR, cornerradius=4), showlegend=False,
            text=counts.values, textposition="outside",
            hovertext=[f"{label}: {n} jogos" for label, n in zip(x, counts.values)], hoverinfo="text",
        ), row=1, col=col)
        fig.update_yaxes(title_text="Jogos", rangemode="tozero", range=[0, counts.max() * 1.15], row=1, col=col)
    fig.update_layout(
        title="Distribuição dos 1119 jogos no calendário (a partir de data_jogo)",
        template="plotly_white", height=420, separators=",.", bargap=0.3,
    )
    return fig
