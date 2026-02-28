from pathlib import Path
import pandas as pd

from src.config import DEFAULT_RAW_FILE
from src.logging_utils import get_logger

logger = get_logger(__name__)

def load_raw_energy_data(filepath: Path | None = None) -> pd.DataFrame:
    """
    Charge les données brutes d'énergie depuis un CSV.
    """
    if filepath is None:
        filepath = DEFAULT_RAW_FILE

    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath, sep=",", encoding="utf-8")
    logger.info(f"Loaded {len(df)} rows")
    return df