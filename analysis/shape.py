import pandas as pd

MODERATE = 0.5
STRONG = 1.0
SMALL_MEAN = 10


def skewness(s: pd.Series) -> float:
    return s.skew()


def direction(g: float) -> str:
    if g > 0:
        return "à direita (positiva)"
    elif g < 0:
        return "à esquerda (negativa)"
    else:
        return "simétrica"


def intensity(g: float) -> str:
    if abs(g) < MODERATE:
        return "aproximadamente simétrica"
    elif abs(g) < STRONG:
        return "moderada"
    else:
        return "forte"


def cv(s: pd.Series) -> float:
    return s.std() / s.mean() * 100


def poisson_cv(s: pd.Series) -> float:
    return 100 / s.mean() ** 0.5


def robust_cv(s: pd.Series) -> float:
    q1, q3 = s.quantile([0.25, 0.75])
    return (q3 - q1) / s.median() * 100


def by_team(df: pd.DataFrame, teams: list[str], columns: list[str], func) -> pd.DataFrame:
    games = df[df["time"].isin(teams)]
    return games.groupby("time")[columns].agg(func).T[teams]


def skew_table(df: pd.DataFrame, teams: list[str], columns: list[str]) -> pd.DataFrame:
    return by_team(df, teams, columns, skewness)


def cv_table(df: pd.DataFrame, teams: list[str], columns: list[str]) -> pd.DataFrame:
    return by_team(df, teams, columns, cv)


def robust_cv_table(df: pd.DataFrame, teams: list[str], columns: list[str]) -> pd.DataFrame:
    return by_team(df, teams, columns, robust_cv)


def poisson_cv_table(df: pd.DataFrame, teams: list[str], columns: list[str]) -> pd.DataFrame:
    return by_team(df, teams, columns, poisson_cv)
