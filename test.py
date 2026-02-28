import sys
from pathlib import Path

# Permet d'importer le code depuis src/
PROJECT_ROOT = Path.cwd().parents[0] if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(PROJECT_ROOT / "src"))

from src.ingestion.load_data import load_raw_energy_data
from src.preprocessing.clean_data import reshape_energy_data
from src.analysis.descriptive_stats import compute_basic_kpis, yearly_aggregates

print("Imports OK")
df_raw = load_raw_energy_data()
df_energy = reshape_energy_data(df_raw)

kpis = compute_basic_kpis(df_energy)
print("KPI globaux :", kpis)

df_yearly = yearly_aggregates(df_energy)
print(df_yearly)

from src.modeling.simple_trend import fit_linear_trend, predict_future

model = fit_linear_trend(df_yearly, "conso_energie")

future = predict_future(model, [2018, 2019, 2020])
print(future)

from src.monitoring.data_quality import data_quality_report

dq = data_quality_report(df_energy)
print(dq)
