import pandas as pd
from src.logging_utils import get_logger

logger = get_logger(__name__)

def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne un rapport de qualité de données :
    - types
    - valeurs manquantes
    - ratio de manquants
    - valeurs négatives
    """
    logger.info("Generating data quality report")

    report = pd.DataFrame({
        "dtype": df.dtypes,
        "missing_count": df.isna().sum(),
        "missing_ratio": df.isna().mean(),
        "negative_count": (df.select_dtypes(include="number") < 0).sum(),
    })

    return report