import pandas as pd
from analysis.outliers import IQR_K, iqr_outliers

MODERATE = 0.5
STRONG = 1.0
SMALL_MEAN = 10
IQR_EXTREME = 3
# referências numa distribuição normal: desvio = IQR ÷ 1,349; amplitude de ~75 jogos ≈ 3,57 IQR (simulação)
NORMAL_STD_IQR = 1 / 1.349
NORMAL_RANGE_IQR = 3.57


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


def iqr(s: pd.Series) -> float:
    q1, q3 = s.quantile([0.25, 0.75])
    return q3 - q1


def cv(s: pd.Series) -> float:
    return s.std() / s.mean() * 100


def poisson_cv(s: pd.Series) -> float:
    return 100 / s.mean() ** 0.5


def robust_cv(s: pd.Series) -> float:
    return iqr(s) / s.median() * 100


def by_team(df: pd.DataFrame, teams: list[str], columns: list[str], func) -> pd.DataFrame:
    games = df[df["time"].isin(teams)]
    return games.groupby("time")[columns].agg(func).T[teams]


def iqr_table(df: pd.DataFrame, teams: list[str], columns: list[str]) -> pd.DataFrame:
    return by_team(df, teams, columns, iqr)


def skew_table(df: pd.DataFrame, teams: list[str], columns: list[str]) -> pd.DataFrame:
    return by_team(df, teams, columns, skewness)


def cv_table(df: pd.DataFrame, teams: list[str], columns: list[str]) -> pd.DataFrame:
    return by_team(df, teams, columns, cv)


def robust_cv_table(df: pd.DataFrame, teams: list[str], columns: list[str]) -> pd.DataFrame:
    return by_team(df, teams, columns, robust_cv)


def poisson_cv_table(df: pd.DataFrame, teams: list[str], columns: list[str]) -> pd.DataFrame:
    return by_team(df, teams, columns, poisson_cv)


def robustness(s: pd.Series) -> pd.DataFrame:
    kept = s[~iqr_outliers(s)]
    return pd.DataFrame({
        "com_outliers": [s.std(), iqr(s), s.max() - s.min()],
        "sem_outliers": [kept.std(), iqr(kept), kept.max() - kept.min()],
    }, index=["desvio_padrao", "iqr", "amplitude"])


def tukey_hinges(s: pd.Series) -> tuple[float, float]:
    ordered = s.sort_values().reset_index(drop=True)
    half = (len(ordered) + 1) // 2
    return ordered.iloc[:half].median(), ordered.iloc[-half:].median()


def outlier_rule(s: pd.Series, q1: float, q3: float, k: float) -> dict:
    spread = q3 - q1
    low, high = q1 - k * spread, q3 + k * spread
    return {"q1": q1, "q3": q3, "iqr": spread, "limite_inferior": low, "limite_superior": high,
            "abaixo": int((s < low).sum()), "acima": int((s > high).sum())}


def rule_comparison(s: pd.Series) -> pd.DataFrame:
    q1, q3 = s.quantile([0.25, 0.75])
    return pd.DataFrame({
        "Interpolação linear, k = 1,5 (adotada)": outlier_rule(s, q1, q3, IQR_K),
        "Interpolação linear, k = 3": outlier_rule(s, q1, q3, IQR_EXTREME),
        "Dobradiças de Tukey, k = 1,5": outlier_rule(s, *tukey_hinges(s), IQR_K),
    }).T
