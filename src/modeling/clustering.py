import pandas as pd
from sklearn.cluster import KMeans
from src.logging_utils import get_logger

logger = get_logger(__name__)


def cluster_collectivities(
    df: pd.DataFrame, n_clusters: int = 3, features: list[str] | None = None
) -> tuple[pd.DataFrame, KMeans]:
    """Partitionne les collectivités en `n_clusters` groupes à l'aide de KMeans.

    Le DataFrame retourné contient une colonne ``cluster``.

    Parameters
    ----------
    df : pd.DataFrame
        Table de base contenant au moins les colonnes indiquées dans ``features``.
    n_clusters : int
        Nombre de clusters demandés.
    features : list[str] | None
        Colonnes numériques à utiliser pour le clustering. Si ``None`` on prend
        ``["conso_energie", "depense_energie"]``.

    Returns
    -------
    (pd.DataFrame, KMeans)
        DataFrame étendu et objet KMeans entraîné (pour réaffecter ultérieurement).
    """
    logger.info(f"Clustering collectivités en {n_clusters} groupes")
    if features is None:
        features = ["conso_energie", "depense_energie"]

    X = df[features].fillna(0).values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    df_out = df.copy()
    df_out["cluster"] = labels
    return df_out, kmeans


def assign_clusters(df: pd.DataFrame, kmeans_model: KMeans, features: list[str]) -> pd.Series:
    """Assigne des observations à des clusters à l'aide d'un modèle KMeans pré-entrainé."""
    X = df[features].fillna(0).values
    return pd.Series(kmeans_model.predict(X), index=df.index)
