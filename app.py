import pandas as pd
from flask import Flask, abort, render_template, request

from analysis.charts import (VARIABLE_NAMES, boxplot, cv_bars, date_bars, histogram_kde, iqr_ranges, mean_bars,
                             method_heatmap, method_strip, team_dot, team_palette, votes_bars, zscore_bars)
from analysis.cleaning import (DERIVED_COLUMNS, ML_EXCLUDED, clean, convert_dates, derive_matchups, games_by_month,
                               games_by_weekday, ml_features, rename_columns)
from analysis.descriptive import compare_teams, league_zscores, outlier_games, team_record, team_table
from analysis.dictionary import COLUMNS, data_dictionary, games_per_team, team_info
from analysis.formatting import format_measures, number
from analysis import outliers as od
from analysis.loader import load_raw
from analysis.quality import (DOMAIN, ZERO_MEANING, duplicates, inconsistencies_by_column, inconsistency_checks,
                              missing_counts, out_of_domain, special_values, validate_percentages)
from analysis.shape import (IQR_EXTREME, MODERATE, NORMAL_RANGE_IQR, NORMAL_STD_IQR, SMALL_MEAN, STRONG, by_team, cv,
                            cv_table, direction, intensity, iqr, iqr_table, poisson_cv, robust_cv_table, robustness,
                            rule_comparison, skew_table)
from config import GROUP_TEAMS, VARIABLES

PENDING_SECTIONS = {
}


def team_options(df) -> list[str]:
    others = sorted(set(df["time"]) - set(GROUP_TEAMS))
    return GROUP_TEAMS + others


PERCENT_FORMULAS = {
    "aproveitamento_arremessos": "arremessos convertidos ÷ tentados",
    "aproveitamento_3pts": "cestas de 3 convertidas ÷ tentadas",
    "aproveitamento_lances_livres": "lances livres convertidos ÷ tentados",
    "aproveitamento_acumulado": "vitórias ÷ (vitórias + derrotas) acumuladas",
}


def domain_text(low, high) -> str:
    return f"≥ {low}" if high == float("inf") else f"{low} a {high}"


def choose(value, options, default):
    return value if value in options else default


def outliers_elsewhere(df, team: str, variable: str) -> list[str]:
    counts = team_table(df, team, VARIABLES)["outliers"]
    return [f"{VARIABLE_NAMES[col]} ({n})" for col, n in counts.items() if n > 0 and col != variable]


LEAGUE = "liga"
LOWER_IS_BETTER = {"faltas"}
# decoração da página inicial: fotos oficiais da NBA (ID do jogador em nba.com/player/<id>)
HEADSHOT_URL = "https://cdn.nba.com/headshots/nba/latest/1040x760/{}.png"
FEATURED_PLAYERS = [
    {"nome": "Donovan Mitchell", "time": "Cleveland Cavaliers", "id": 1628378, "lado": "left"},
    {"nome": "Jalen Brunson", "time": "New York Knicks", "id": 1628973, "lado": "right"},
]
OUTLIER_PARAMS = {
    "z": od.Z_THRESHOLD, "iqr_k": od.IQR_K, "grubbs_alpha": od.GRUBBS_ALPHA,
    "contamination": od.CONTAMINATION, "knn_k": od.KNN_NEIGHBORS, "lof_k": od.LOF_NEIGHBORS,
    "eps": od.DBSCAN_EPS, "min_samples": 2 * len(VARIABLES),
}


def flagged_games(df, joint, rows, min_votes: int) -> list[dict]:
    selected = rows & (joint["votos"] >= min_votes)
    methods = joint.loc[selected, list(od.MULTIVARIATE_METHODS)]
    games = df.loc[selected, ["data_jogo", "confronto", "minutos"] + VARIABLES].copy()
    games["votos"] = joint.loc[selected, "votos"]
    games["metodos"] = [", ".join(methods.columns[row]) for row in methods.to_numpy()]
    return games.sort_values(["votos", "data_jogo"], ascending=[False, True]).to_dict("records")


def rank_record(record) -> pd.DataFrame:
    return pd.DataFrame({
        "aproveitamento": record["aproveitamento"].rank(ascending=False, method="min"),
        "pontos": record["pontos"].rank(ascending=False, method="min"),
        "sofridos": record["sofridos"].rank(ascending=True, method="min"),
        "saldo": record["saldo"].rank(ascending=False, method="min"),
    }).astype(int)


def cumulative_matches(df) -> float:
    ordered = df.sort_values(["time", "data_jogo"])
    wins = ordered.groupby("time")["vitoria"].cumsum()
    losses = ordered.groupby("time")["vitoria"].transform(lambda s: (1 - s).cumsum())
    same = (ordered["vitorias_acumuladas"] == wins) & (ordered["derrotas_acumuladas"] == losses)
    return same.mean() * 100


def create_app() -> Flask:
    app = Flask(__name__)
    raw = load_raw()
    df = clean(raw)
    before_fix = derive_matchups(rename_columns(convert_dates(raw)))
    teams = team_options(df)
    column_flags = od.all_column_flags(df, VARIABLES)
    joint = od.joint_flags(df, VARIABLES)
    inconsistent = inconsistencies_by_column(inconsistency_checks(df)).reindex(VARIABLES, fill_value=0)
    cv_ranks = cv_table(df, teams, VARIABLES).rank(axis=1).astype(int)
    info = team_info(df)
    record = team_record(df)
    record_ranks = rank_record(record)
    means = df.groupby("time")[VARIABLES].mean()
    mean_ranks = means.rank(ascending=False, method="min").astype(int)

    @app.route("/")
    def index():
        return render_template(
            "inicio.html",
            info=info, main=GROUP_TEAMS, record=record, record_ranks=record_ranks,
            main_logos={team: info.loc[info["time"] == team, "logo"].iloc[0] for team in GROUP_TEAMS},
            main_siglas={team: info.index[info["time"] == team][0] for team in GROUP_TEAMS},
            rows=len(df), games=df["id_jogo"].nunique(),
            first=df["data_jogo"].min(), last=df["data_jogo"].max(), number=number,
            players=[{**player, "foto": HEADSHOT_URL.format(player["id"])} for player in FEATURED_PLAYERS],
        )

    @app.route("/time/<sigla>")
    def team_page(sigla):
        sigla = sigla.upper()
        if sigla not in info.index:
            abort(404)
        team = info.loc[sigla, "time"]
        partner = GROUP_TEAMS[1] if team == GROUP_TEAMS[0] else GROUP_TEAMS[0]
        z = league_zscores(df, [team], VARIABLES).loc[team]
        performance = z * pd.Series({col: -1 if col in LOWER_IS_BETTER else 1 for col in VARIABLES})
        chart = zscore_bars(df, [team], VARIABLES).to_html(full_html=False, include_plotlyjs="cdn")
        return render_template(
            "time.html",
            team=team, sigla=sigla, logo=info.loc[sigla, "logo"], main=GROUP_TEAMS, partner=partner,
            record=record.loc[team], ranks=record_ranks.loc[team], teams_count=len(info),
            variable_names=VARIABLE_NAMES, means=means.loc[team], league_means=means.mean(),
            mean_ranks=mean_ranks.loc[team], z=z, best=performance.idxmax(), worst=performance.idxmin(),
            all_above=bool((performance >= 0).all()),
            cv=cv_table(df, [team], VARIABLES)[team], iqr=iqr_table(df, [team], VARIABLES)[team],
            skew=skew_table(df, [team], VARIABLES)[team], outliers=team_table(df, team, VARIABLES)["outliers"],
            chart=chart, number=number,
        )

    @app.route("/descritiva")
    def descritiva():
        team_a = choose(request.args.get("time_a"), teams, GROUP_TEAMS[0])
        team_b = choose(request.args.get("time_b"), teams, GROUP_TEAMS[1])
        variable = choose(request.args.get("variavel"), VARIABLES, VARIABLES[0])
        selected = [team_a] if team_a == team_b else [team_a, team_b]

        table = format_measures(compare_teams(df, selected, variable))
        chart = boxplot(df, selected, variable).to_html(full_html=False, include_plotlyjs="cdn")
        means_chart = mean_bars(df, selected, VARIABLES).to_html(full_html=False, include_plotlyjs=False)
        z_chart = zscore_bars(df, selected, VARIABLES).to_html(full_html=False, include_plotlyjs=False)
        colors = {team: team_dot(c) for team, c in team_palette(selected).items()}
        outliers = {team: outlier_games(df, team, variable).to_dict("records") for team in selected}
        elsewhere = {team: outliers_elsewhere(df, team, variable) for team in selected}

        return render_template(
            "descritiva.html",
            teams=teams, team_a=team_a, team_b=team_b, variable=variable,
            variable_names=VARIABLE_NAMES, colors=colors,
            table=table, chart=chart, outliers=outliers, elsewhere=elsewhere,
            means_chart=means_chart, z_chart=z_chart,
        )

    @app.route("/qualidade")
    def qualidade():
        special = special_values(df)
        domain = pd.DataFrame({
            "dominio": {col: domain_text(low, high) for col, (low, high) in DOMAIN.items()},
            "antes": out_of_domain(before_fix),
            "depois": out_of_domain(df),
        })
        validation = pd.DataFrame({
            "formula": PERCENT_FORMULAS,
            "antes": validate_percentages(before_fix),
            "depois": validate_percentages(df),
        })
        missing = missing_counts(df)
        return render_template(
            "qualidade.html",
            numeric_columns=len(special),
            zeros=special[(special["zeros"] > 0) | (special["negativos"] > 0)],
            zero_meaning=ZERO_MEANING,
            dups=duplicates(df), games=df["id_jogo"].nunique(), rows=len(df),
            domain=domain, validation=validation,
            missing=missing, missing_total=int(missing.sum()),
            checks=inconsistency_checks(df),
        )

    @app.route("/outliers")
    def outliers():
        scope = choose(request.args.get("escopo"), [LEAGUE] + teams, LEAGUE)
        variable = choose(request.args.get("variavel"), VARIABLES, VARIABLES[0])
        rows = df["time"].notna() if scope == LEAGUE else df["time"] == scope
        scope_name = "liga inteira (30 times)" if scope == LEAGUE else scope

        counts = od.counts_from_flags(column_flags, rows)
        table = counts.assign(inconsistencias=inconsistent).rename(index=VARIABLE_NAMES)
        heatmap = method_heatmap(counts, scope_name).to_html(full_html=False, include_plotlyjs="cdn")
        strip = method_strip(df, column_flags[variable], variable, rows, scope_name).to_html(
            full_html=False, include_plotlyjs=False)
        votes = votes_bars(joint.loc[rows, "votos"], scope_name).to_html(full_html=False, include_plotlyjs=False)
        min_votes = 4 if scope == LEAGUE else 1

        return render_template(
            "outliers.html",
            teams=teams, scope=scope, scope_name=scope_name, league=LEAGUE, variable=variable,
            variable_names=VARIABLE_NAMES, rows=int(rows.sum()), params=OUTLIER_PARAMS,
            table=table, heatmap=heatmap, strip=strip, votes=votes,
            marked=int((joint.loc[rows, "votos"] > 0).sum()),
            games=flagged_games(df, joint, rows, min_votes), min_votes=min_votes,
        )

    @app.route("/assimetria")
    def assimetria():
        team_a = choose(request.args.get("time_a"), teams, GROUP_TEAMS[0])
        team_b = choose(request.args.get("time_b"), teams, GROUP_TEAMS[1])
        variable = choose(request.args.get("variavel"), VARIABLES, VARIABLES[0])
        selected = [team_a] if team_a == team_b else [team_a, team_b]

        hist = histogram_kde(df, selected, variable).to_html(full_html=False, include_plotlyjs="cdn")
        box = boxplot(df, selected, variable).to_html(full_html=False, include_plotlyjs=False)
        colors = {team: team_dot(c) for team, c in team_palette(selected).items()}

        return render_template(
            "assimetria.html",
            teams=teams, team_a=team_a, team_b=team_b, variable=variable,
            variable_names=VARIABLE_NAMES, colors=colors,
            skew=skew_table(df, selected, VARIABLES), league=df[VARIABLES].skew(),
            measures=compare_teams(df, selected, variable), hist=hist, box=box,
            direction=direction, intensity=intensity, number=number,
            moderate=MODERATE, strong=STRONG,
        )

    @app.route("/cv")
    def coeficiente_variacao():
        team_a = choose(request.args.get("time_a"), teams, GROUP_TEAMS[0])
        team_b = choose(request.args.get("time_b"), teams, GROUP_TEAMS[1])
        selected = [team_a] if team_a == team_b else [team_a, team_b]

        chart = cv_bars(df, selected, VARIABLES).to_html(full_html=False, include_plotlyjs="cdn")
        colors = {team: team_dot(c) for team, c in team_palette(selected).items()}

        return render_template(
            "cv.html",
            teams=teams, team_a=team_a, team_b=team_b, selected=selected,
            variable_names=VARIABLE_NAMES, colors=colors, chart=chart,
            cv=cv_table(df, selected, VARIABLES), ranks=cv_ranks[selected],
            robust=robust_cv_table(df, selected, VARIABLES), skew=skew_table(df, selected, VARIABLES),
            std=by_team(df, selected, VARIABLES, "std"), means=by_team(df, selected, VARIABLES, "mean"),
            league_cv=df[VARIABLES].agg(cv), league_mean=df[VARIABLES].mean(),
            league_poisson=df[VARIABLES].agg(poisson_cv),
            small_mean=SMALL_MEAN, number=number,
        )

    @app.route("/iqr")
    def intervalo_interquartil():
        team_a = choose(request.args.get("time_a"), teams, GROUP_TEAMS[0])
        team_b = choose(request.args.get("time_b"), teams, GROUP_TEAMS[1])
        variable = choose(request.args.get("variavel"), VARIABLES, VARIABLES[0])
        selected = [team_a] if team_a == team_b else [team_a, team_b]
        series = {team: df[df["time"] == team][variable] for team in selected}

        ranges = iqr_ranges(df, selected, VARIABLES).to_html(full_html=False, include_plotlyjs="cdn")
        box = boxplot(df, selected, variable).to_html(full_html=False, include_plotlyjs=False)
        colors = {team: team_dot(c) for team, c in team_palette(selected).items()}

        return render_template(
            "iqr.html",
            teams=teams, team_a=team_a, team_b=team_b, variable=variable, selected=selected,
            variable_names=VARIABLE_NAMES, colors=colors, ranges=ranges, box=box,
            q1=by_team(df, selected, VARIABLES, lambda s: s.quantile(0.25)),
            q3=by_team(df, selected, VARIABLES, lambda s: s.quantile(0.75)),
            iqr=iqr_table(df, selected, VARIABLES), league_iqr=df[VARIABLES].agg(iqr),
            amplitude=by_team(df, selected, VARIABLES, lambda s: s.max() - s.min()),
            std=by_team(df, selected, VARIABLES, "std"),
            cv=cv_table(df, selected, VARIABLES), skew=skew_table(df, selected, VARIABLES),
            robust={team: robustness(s) for team, s in series.items()},
            rules={team: rule_comparison(s) for team, s in series.items()},
            normal_std=NORMAL_STD_IQR, normal_range=NORMAL_RANGE_IQR, extreme=IQR_EXTREME, number=number,
        )

    @app.route("/dados")
    def dados():
        dictionary = data_dictionary(raw)
        per_team = games_per_team(df)
        return render_template(
            "dados.html",
            rows=len(raw), columns=raw.shape[1], teams_count=df["time"].nunique(), games=df["id_jogo"].nunique(),
            first=df["data_jogo"].min(), last=df["data_jogo"].max(),
            per_team=per_team, per_team_counts=per_team.value_counts().sort_index(),
            dictionary=dictionary,
            concept_counts=dictionary["tipo"].value_counts(), pandas_counts=dictionary["tipo_pandas"].value_counts(),
        )

    @app.route("/tratamento")
    def tratamento():
        dates = date_bars(games_by_month(df), games_by_weekday(df)).to_html(full_html=False, include_plotlyjs="cdn")
        return render_template(
            "tratamento.html",
            raw_date=raw["GAME_DATE"].iloc[0], raw_date_type=str(raw["GAME_DATE"].dtype),
            clean_date_type=str(df["data_jogo"].dtype), first=df["data_jogo"].min(), last=df["data_jogo"].max(),
            dates=dates, renames=COLUMNS,
            wl_values=sorted(df["vitoria"].unique()), cumulative=cumulative_matches(df),
            max_wins=df["vitorias_acumuladas"].max(), max_losses=df["derrotas_acumuladas"].max(),
            home=int(df["casa"].sum()), away=int((df["casa"] == 0).sum()), opponents=df["adversario"].nunique(),
            examples=df[["confronto", "casa", "adversario"]].drop_duplicates("casa").to_dict("records"),
            derived=DERIVED_COLUMNS, excluded=ML_EXCLUDED,
            raw_shape=raw.shape, clean_shape=df.shape, ml_shape=ml_features(df).shape,
        )

    @app.route("/sintese")
    def sintese():
        return render_template(
            "sintese.html",
            main=GROUP_TEAMS, variable_names=VARIABLE_NAMES,
            colors={team: team_dot(c) for team, c in team_palette(GROUP_TEAMS).items()},
            record=record.loc[GROUP_TEAMS], record_ranks=record_ranks.loc[GROUP_TEAMS],
            league_home=df.loc[df["casa"] == 1, "vitoria"].mean(),
            means=means.loc[GROUP_TEAMS].T, league_means=means.mean(), mean_ranks=mean_ranks.loc[GROUP_TEAMS].T,
            z=league_zscores(df, GROUP_TEAMS, VARIABLES).T,
            cv=cv_table(df, GROUP_TEAMS, VARIABLES), iqr=iqr_table(df, GROUP_TEAMS, VARIABLES),
            skew=skew_table(df, GROUP_TEAMS, VARIABLES),
            outliers=pd.DataFrame({team: team_table(df, team, VARIABLES)["outliers"] for team in GROUP_TEAMS}),
            number=number,
        )

    @app.route("/<section>")
    def pending(section):
        if section not in PENDING_SECTIONS:
            abort(404)
        return render_template("em_construcao.html", title=PENDING_SECTIONS[section])

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
