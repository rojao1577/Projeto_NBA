import pandas as pd
from config import DECIMALS

MEASURE_NAMES = {
    "jogos": "Jogos",
    "media": "Média",
    "mediana": "Mediana",
    "moda": "Moda",
    "minimo": "Mínimo",
    "maximo": "Máximo",
    "amplitude": "Amplitude",
    "desvio_padrao": "Desvio padrão",
    "variancia": "Variância",
    "q1": "Q1",
    "q3": "Q3",
    "iqr": "IQR",
    "limite_inferior": "Limite inferior",
    "limite_superior": "Limite superior",
    "outliers": "Outliers (IQR)",
}
COUNT_MEASURES = {"jogos", "outliers"}


def number(value: float, decimals: int = DECIMALS) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def format_measure(measure: str, value) -> str:
    if measure in COUNT_MEASURES:
        return str(int(value))
    if measure == "moda":
        return " e ".join(number(v) for v in value)
    return number(value)


def format_measures(table: pd.DataFrame) -> pd.DataFrame:
    formatted = pd.DataFrame(
        {team: [format_measure(m, v) for m, v in table[team].items()] for team in table.columns},
        index=table.index,
    )
    return formatted.rename(index=MEASURE_NAMES)