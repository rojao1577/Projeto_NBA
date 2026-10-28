import numpy as np
import pandas as pd

DOMAIN = {
    "aproveitamento_arremessos": (0, 1),
    "aproveitamento_3pts": (0, 1),
    "aproveitamento_lances_livres": (0, 1),
    "aproveitamento_acumulado": (0, 1),
    "vitoria": (0, 1),
    "casa": (0, 1),
    "minutos": (240, np.inf),
}


def special_values(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes("number")
    return pd.DataFrame({
        "zeros": (num == 0).sum(),
        "negativos": (num < 0).sum(),
    })

def duplicates(df: pd.DataFrame) -> dict:
    return {
        "linhas_identicas": int(df.duplicated().sum()),
        "time_jogo_repetido": int(df.duplicated(["id_time", "id_jogo"]).sum()),
    }

def out_of_domain(df: pd.DataFrame) -> pd.Series:
    counts = {}
    for col, (low, high) in DOMAIN.items():
        counts[col] = int((~df[col].between(low, high)).sum())
    return pd.Series(counts)    


def validate_percentages(df: pd.DataFrame) -> pd.Series:
    recalculated = {
        "aproveitamento_arremessos": df["arremessos_convertidos"] / df["arremessos_tentados"],
        "aproveitamento_3pts": df["cestas_3_convertidas"] / df["cestas_3_tentadas"],
        "aproveitamento_lances_livres": df["lances_livres_convertidos"] / df["lances_livres_tentados"],
        "aproveitamento_acumulado": df["vitorias_acumuladas"]
        / (df["vitorias_acumuladas"] + df["derrotas_acumuladas"]),
    }
    counts = {}
    for col, real in recalculated.items():
        diff = (df[col] - real).abs()
        counts[col] = int((diff > 0.001).sum())
    return pd.Series(counts)


def missing_counts(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum()