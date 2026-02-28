import pandas as pd
from src.logging_utils import get_logger

logger = get_logger(__name__)

def compute_basic_kpis(df: pd.DataFrame) -> dict:
    """
    Calcule quelques KPI globaux simples (sans CO2).
    """
    logger.info("Computing basic KPIs")

    kpis = {
        "total_conso": df["conso_energie"].sum(),
        "total_depense": df["depense_energie"].sum(),
    }

    return kpis


def yearly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège par année les indicateurs principaux (sans CO2).
    """
    logger.info("Computing yearly aggregates")

    agg = (
        df.groupby("annee")[["conso_energie", "depense_energie"]]
        .sum()
        .reset_index()
    )
    return agg
