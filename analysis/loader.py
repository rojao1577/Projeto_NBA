import pandas as pd

from config import RAW_CSV

from functools import lru_cache


@lru_cache(maxsize=1)
def _read_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV)


def load_raw() -> pd.DataFrame:
    return _read_raw().copy()