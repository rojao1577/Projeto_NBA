import pandas as pd
import numpy as np
from scipy import stats
from pyod.models.ecod import ECOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.cluster import DBSCAN

Z_THRESHOLD = 3
IQR_K = 1.5
GRUBBS_ALPHA = 0.05
CONTAMINATION = 0.01
KNN_NEIGHBORS = 5
LOF_NEIGHBORS = 20
DBSCAN_EPS = 2.0


def zscore_outliers(s: pd.Series) -> pd.Series:
    z = (s - s.mean()) / s.std()
    return z.abs() > Z_THRESHOLD


def iqr_limits(s: pd.Series) -> tuple[float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - IQR_K * iqr, q3 + IQR_K * iqr


def iqr_outliers(s: pd.Series) -> pd.Series:
    low, high = iqr_limits(s)
    return (s < low) | (s > high)


def grubbs_critical(n: int) -> float:
    t = stats.t.ppf(1 - GRUBBS_ALPHA / (2 * n), n - 2)
    return (n - 1) / np.sqrt(n) * np.sqrt(t**2 / (n - 2 + t**2))


def grubbs_outliers(s: pd.Series) -> pd.Series:
    remaining = s.copy()
    outliers = pd.Series(False, index=s.index)
    while len(remaining) > 2:
        distance = (remaining - remaining.mean()).abs() / remaining.std()
        suspect = distance.idxmax()
        if distance[suspect] <= grubbs_critical(len(remaining)):
            break
        outliers[suspect] = True
        remaining = remaining.drop(suspect)
    return outliers


def ecod_outliers(X: pd.DataFrame) -> pd.Series:
    model = ECOD(contamination=CONTAMINATION).fit(X)
    return pd.Series(model.labels_ == 1, index=X.index)


def standardize(X: pd.DataFrame) -> pd.DataFrame:
    return (X - X.mean()) / X.std()


def knn_outliers(X: pd.DataFrame) -> pd.Series:
    model = KNN(n_neighbors=KNN_NEIGHBORS, contamination=CONTAMINATION).fit(standardize(X))
    return pd.Series(model.labels_ == 1, index=X.index)


def lof_outliers(X: pd.DataFrame) -> pd.Series:
    model = LOF(n_neighbors=LOF_NEIGHBORS, contamination=CONTAMINATION).fit(standardize(X))
    return pd.Series(model.labels_ == 1, index=X.index)


def dbscan_outliers(X: pd.DataFrame) -> pd.Series:
    model = DBSCAN(eps=DBSCAN_EPS, min_samples=2 * X.shape[1]).fit(standardize(X))
    return pd.Series(model.labels_ == -1, index=X.index)


UNIVARIATE_METHODS = {
    "Z-score": zscore_outliers,
    "IQR": iqr_outliers,
    "Grubbs": grubbs_outliers,
}

MULTIVARIATE_METHODS = {
    "ECOD": ecod_outliers,
    "K-NN": knn_outliers,
    "LOF": lof_outliers,
    "DBSCAN": dbscan_outliers,
}


def column_flags(df: pd.DataFrame, col: str) -> pd.DataFrame:
    flags = {name: method(df[col]) for name, method in UNIVARIATE_METHODS.items()}
    flags |= {name: method(df[[col]]) for name, method in MULTIVARIATE_METHODS.items()}
    return pd.DataFrame(flags)


def outlier_counts(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({col: column_flags(df, col).sum() for col in columns}).T


def joint_flags(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    flags = pd.DataFrame({name: method(df[columns]) for name, method in MULTIVARIATE_METHODS.items()})
    flags["votos"] = flags.sum(axis=1)
    return flags