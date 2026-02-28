import pandas as pd
from src.logging_utils import get_logger

logger = get_logger(__name__)

def reshape_energy_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme le fichier EPC en table énergie standardisée (format long).
    """
    logger.info("Reshaping energy data into long format")

    ENERGY_COLUMNS = {
        "fuel_domestique": ("conso_fuel_domestique", "depenses_fuel_domestique"),
        "fuel_lourd": ("conso_fuel_lourd", "depenses_fuel_lourd"),
        "gaz_naturel": ("conso_gaz_naturel", "depenses_gaz_naturel"),
        "charbon": ("conso_charbon", "depenses_charbon"),
        "butane": ("conso_butane", "depenses_butane"),
        "bois": ("conso_bois", "depenses_bois"),
        "rdc": ("conso_rdc", "depenses_rdc"),
        "electricite": ("conso_electricite", "depenses_electricite"),
        "eclairage_public": ("conso_eclairage_public", "depenses_eclairage_public"),
    }

    rows = []

    for _, row in df.iterrows():
        identifiant = row["identifiant"]

        for energy_type, (conso_col, dep_col) in ENERGY_COLUMNS.items():
            conso = row.get(conso_col, None)
            dep = row.get(dep_col, None)

            rows.append({
                "identifiant": identifiant,
                "annee": 2017,
                "type_energie": energy_type,
                "conso_energie": conso,
                "depense_energie": dep,
            })

    df_energy = pd.DataFrame(rows)

    df_energy["conso_energie"] = pd.to_numeric(df_energy["conso_energie"], errors="coerce")
    df_energy["depense_energie"] = pd.to_numeric(df_energy["depense_energie"], errors="coerce")

    return df_energy