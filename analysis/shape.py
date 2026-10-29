import pandas as pd

MODERATE = 0.5
STRONG = 1.0


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


def skew_table(df: pd.DataFrame, teams: list[str], columns: list[str]) -> pd.DataFrame:
    games = df[df["time"].isin(teams)]
    table = games.groupby("time")[columns].skew().T
    return table[teams]