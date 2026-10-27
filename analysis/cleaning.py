import pandas as pd

from analysis.dictionary import RENAME_MAP


ML_EXCLUDED = {
    "id_time": "redundante com time (mesma informação em forma de número)",
    "id_jogo": "identificador do jogo; não diz nada sobre o desempenho",
    "vitorias_acumuladas": "vazamento de dados: já inclui o resultado do jogo",
    "derrotas_acumuladas": "vazamento de dados: já inclui o resultado do jogo",
    "aproveitamento_acumulado": "vazamento de dados: calculado com o resultado do jogo",
    "confronto": "redundante: informação extraída para casa e adversario",
    "data_jogo": "data bruta não generaliza; informação extraída para mes e dia_semana",
}


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%Y-%m-%d")
    df["mes"] = df["GAME_DATE"].dt.month
    df["dia_semana"] = df["GAME_DATE"].dt.dayofweek
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=RENAME_MAP)


def derive_matchups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["casa"] = df["confronto"].str.contains(" vs. ", regex=False).astype(int)
    df["adversario"] = df["confronto"].str.split().str[-1]
    return df


def ml_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=list(ML_EXCLUDED))