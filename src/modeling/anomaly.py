import pandas as pd
from sklearn.ensemble import IsolationForest
from src.logging_utils import get_logger

logger = get_logger(__name__)


def detect_anomalies_isolation(
    df: pd.DataFrame, features: list[str], contamination: float = 0.05
) -> tuple[pd.DataFrame, IsolationForest]:
    """Repère les anomalies avec un IsolationForest.

    La colonne ``anomaly`` (booléen) est ajoutée au DataFrame retourné.

    Parameters
    ----------
    df : pd.DataFrame
        Jeu de données contenant les variables listées dans ``features``.
    features : list[str]
        Colonnes numériques sur lesquelles s'entraîner.
    contamination : float
        Fraction attendue d'anomalies (paramètre du modèle).

    Returns
    -------
    (pd.DataFrame, IsolationForest)
        DataFrame contenant la colonne ``anomaly`` et l'objet IsolationForest entrainé.
    """
    logger.info("Détection d'anomalies (IsolationForest)")
    iso = IsolationForest(contamination=contamination, random_state=42)
    X = df[features].fillna(0).values
    preds = iso.fit_predict(X)
    df_out = df.copy()
    df_out["anomaly"] = preds == -1
    return df_out, iso


def detect_anomalies_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.DataFrame:
    """Détection d'anomalies sur une colonne unique en utilisant le z-score."""
    import numpy as np

    logger.info(f"Détection d'anomalies z-score sur {col}")
    z = (df[col] - df[col].mean()) / df[col].std()
    df_out = df.copy()
    df_out["anomaly"] = z.abs() > threshold
    return df_out
