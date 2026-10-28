from flask import Flask, abort, render_template, request

from analysis.charts import OTHER_COLOR, TEAM_COLORS, VARIABLE_NAMES, boxplot
from analysis.cleaning import clean
from analysis.descriptive import compare_teams, outlier_games, team_table
from analysis.formatting import format_measures
from analysis.loader import load_raw
from config import GROUP_TEAMS, VARIABLES

PENDING_SECTIONS = {
    "dados": "Dados",
    "tratamento": "Tratamento",
    "qualidade": "Qualidade dos dados",
    "outliers": "Outliers",
    "assimetria": "Assimetria",
    "cv": "Coeficiente de variação",
    "iqr": "Intervalo interquartil",
    "sintese": "Síntese",
}


def team_options(df) -> list[str]:
    others = sorted(set(df["time"]) - set(GROUP_TEAMS))
    return GROUP_TEAMS + others


def choose(value, options, default):
    return value if value in options else default


def outliers_elsewhere(df, team: str, variable: str) -> list[str]:
    counts = team_table(df, team, VARIABLES)["outliers"]
    return [f"{VARIABLE_NAMES[col]} ({n})" for col, n in counts.items() if n > 0 and col != variable]


def create_app() -> Flask:
    app = Flask(__name__)
    df = clean(load_raw())
    teams = team_options(df)

    @app.route("/")
    def index():
        return render_template("em_construcao.html", title="Início")

    @app.route("/descritiva")
    def descritiva():
        team_a = choose(request.args.get("time_a"), teams, GROUP_TEAMS[0])
        team_b = choose(request.args.get("time_b"), teams, GROUP_TEAMS[1])
        variable = choose(request.args.get("variavel"), VARIABLES, VARIABLES[0])
        selected = [team_a] if team_a == team_b else [team_a, team_b]

        table = format_measures(compare_teams(df, selected, variable))
        chart = boxplot(df, selected, variable).to_html(full_html=False, include_plotlyjs="cdn")
        outliers = {team: outlier_games(df, team, variable).to_dict("records") for team in selected}
        elsewhere = {team: outliers_elsewhere(df, team, variable) for team in selected}

        return render_template(
            "descritiva.html",
            teams=teams, team_a=team_a, team_b=team_b, variable=variable,
            variable_names=VARIABLE_NAMES, team_colors=TEAM_COLORS, other_color=OTHER_COLOR,
            table=table, chart=chart, outliers=outliers, elsewhere=elsewhere,
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
