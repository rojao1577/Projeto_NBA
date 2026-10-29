import pandas as pd
from analysis.outliers import iqr_limits, iqr_outliers


def describe(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    low, high = iqr_limits(s)
    return pd.Series({
        "jogos": s.count(),
        "media": s.mean(),
        "mediana": s.median(),
        "moda": s.mode().tolist(),
        "minimo": s.min(),
        "maximo": s.max(),
        "amplitude": s.max() - s.min(),
        "desvio_padrao": s.std(),
        "variancia": s.var(),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "limite_inferior": low,
        "limite_superior": high,
        "outliers": int(iqr_outliers(s).sum()),
    })


def team_table(df: pd.DataFrame, team: str, columns: list[str]) -> pd.DataFrame:
    team_games = df[df["time"] == team]
    return pd.DataFrame({col: describe(team_games[col]) for col in columns}).T


def outlier_games(df: pd.DataFrame, team: str, col: str) -> pd.DataFrame:
    team_games = df[df["time"] == team]
    return team_games[iqr_outliers(team_games[col])][["data_jogo", "confronto", "minutos", col]]


def compare_teams(df: pd.DataFrame, teams: list[str], col: str) -> pd.DataFrame:
    return pd.DataFrame({team: describe(df[df["time"] == team][col]) for team in teams})


def league_zscores(df: pd.DataFrame, teams: list[str], columns: list[str]) -> pd.DataFrame:
    means = df.groupby("time")[columns].mean()
    z = (means - means.mean()) / means.std()
    return z.loc[teams]


def points_against(df: pd.DataFrame) -> pd.Series:
    return df.groupby("id_jogo")["pontos"].transform("sum") - df["pontos"]


def team_record(df: pd.DataFrame) -> pd.DataFrame:
    games = df.assign(sofridos=points_against(df))
    record = games.groupby("time").agg(
        jogos=("vitoria", "size"), vitorias=("vitoria", "sum"), aproveitamento=("vitoria", "mean"),
        pontos=("pontos", "mean"), sofridos=("sofridos", "mean"),
    )
    record["derrotas"] = record["jogos"] - record["vitorias"]
    record["saldo"] = record["pontos"] - record["sofridos"]
    record["casa"] = games[games["casa"] == 1].groupby("time")["vitoria"].mean()
    record["fora"] = games[games["casa"] == 0].groupby("time")["vitoria"].mean()
    return record
