import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

DEFAULT_RAW_FILE = DATA_RAW_DIR / "EPC_2_Energie.csv"

print(f"Default raw file path: {DEFAULT_RAW_FILE}")
