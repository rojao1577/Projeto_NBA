from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


RAW_CSV = DATA_DIR / "stats_times_regular_final.csv"


GROUP_TEAMS: list[str] = ["Cleveland Cavaliers", "New York Knicks"]


VARIABLES = [
    "pontos",
    "rebotes",
    "assistencias",
    "roubos",
    "tocos",
    "faltas",
    "cestas_3_convertidas",
]

DECIMALS = 2
