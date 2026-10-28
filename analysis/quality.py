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


ZERO_MEANING = {
    "vitoria": "derrota (0 = perdeu o jogo)",
    "casa": "jogo fora de casa (0 = visitante)",
    "dia_semana": "segunda-feira (0 = segunda, 6 = domingo)",
    "vitorias_acumuladas": "time ainda sem vitória no início da temporada",
    "derrotas_acumuladas": "time ainda sem derrota no início da temporada (o OKC começou 8–0)",
    "aproveitamento_acumulado": "consequência de 0 vitórias acumuladas",
    "tocos": "jogo sem nenhum toco: raro, mas possível",
    "roubos": "jogo sem nenhum roubo de bola: BOS vs. ATL, 27/03/2026",
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

def inconsistency_checks(df: pd.DataFrame) -> pd.DataFrame:
    game = df.groupby("id_jogo")
    opponent_points = game["pontos"].transform("sum") - df["pontos"]
    ordered = df.sort_values(["time", "data_jogo"])
    game_number = ordered.groupby("time").cumcount() + 1

    checks = [
        ("Rebotes = ofensivos + defensivos",
         ["rebotes", "rebotes_ofensivos", "rebotes_defensivos"],
         df["rebotes"] != df["rebotes_ofensivos"] + df["rebotes_defensivos"]),
        ("Pontos = 2 × arremessos convertidos + cestas de 3 + lances livres",
         ["pontos", "arremessos_convertidos", "cestas_3_convertidas", "lances_livres_convertidos"],
         df["pontos"] != 2 * df["arremessos_convertidos"] + df["cestas_3_convertidas"]
         + df["lances_livres_convertidos"]),
        ("Arremessos convertidos ≤ tentados",
         ["arremessos_convertidos", "arremessos_tentados"],
         df["arremessos_convertidos"] > df["arremessos_tentados"]),
        ("Cestas de 3 convertidas ≤ tentadas",
         ["cestas_3_convertidas", "cestas_3_tentadas"],
         df["cestas_3_convertidas"] > df["cestas_3_tentadas"]),
        ("Lances livres convertidos ≤ tentados",
         ["lances_livres_convertidos", "lances_livres_tentados"],
         df["lances_livres_convertidos"] > df["lances_livres_tentados"]),
        ("Cestas de 3 convertidas ≤ arremessos convertidos",
         ["cestas_3_convertidas", "arremessos_convertidos"],
         df["cestas_3_convertidas"] > df["arremessos_convertidos"]),
        ("Minutos = 240 + múltiplo de 25 (prorrogações)",
         ["minutos"],
         (df["minutos"] - 240) % 25 != 0),
        ("Cada jogo tem exatamente 1 vencedor",
         ["vitoria", "id_jogo"],
         game["vitoria"].transform("sum") != 1),
        ("Cada jogo tem exatamente 1 mandante",
         ["casa", "id_jogo"],
         game["casa"].transform("sum") != 1),
        ("Quem venceu fez mais pontos que o adversário",
         ["vitoria", "pontos"],
         (df["vitoria"] == 1) != (df["pontos"] > opponent_points)),
        ("Vitórias + derrotas acumuladas = número do jogo do time",
         ["vitorias_acumuladas", "derrotas_acumuladas"],
         ordered["vitorias_acumuladas"] + ordered["derrotas_acumuladas"] != game_number),
    ]
    return pd.DataFrame(
        [{"regra": rule, "colunas": columns, "inconsistentes": int(mask.sum())}
         for rule, columns, mask in checks]
    )


def inconsistencies_by_column(checks: pd.DataFrame) -> pd.Series:
    return checks.explode("colunas").groupby("colunas")["inconsistentes"].sum()
