import sys
from pathlib import Path

# Permet d'importer le code depuis src/
PROJECT_ROOT = Path.cwd().parents[0] if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(PROJECT_ROOT / "src"))

from src.ingestion.load_data import load_raw_energy_data
from src.preprocessing.clean_data import reshape_energy_data
import numpy as np
from src.analysis.descriptive_stats import compute_basic_kpis, yearly_aggregates

print("Imports OK")
df_raw = load_raw_energy_data()
df_energy = reshape_energy_data(df_raw)

kpis = compute_basic_kpis(df_energy)
print("KPI globaux :", kpis)

df_yearly = yearly_aggregates(df_energy)
print(df_yearly)

# additions for clustering/anomalies
df_by_collectivite = df_energy.groupby("identifiant")[["conso_energie","depense_energie"]].sum().reset_index()
# compute ratio carefully
df_by_collectivite["ratio"] = df_by_collectivite["depense_energie"] / df_by_collectivite["conso_energie"].replace(0, np.nan)
df_by_collectivite["ratio"] = df_by_collectivite["ratio"].fillna(0)

from src.modeling.simple_trend import fit_linear_trend, predict_future
from src.modeling.forecasting import fit_arima, forecast_arima
from src.modeling.clustering import cluster_collectivities
from src.modeling.anomaly import detect_anomalies_isolation, detect_anomalies_zscore

# test simple trend
model = fit_linear_trend(df_yearly, "conso_energie")
future = predict_future(model, [2018, 2019, 2020])
print("Linear trend future:", future)

# test ARIMA forecasting on yearly consumption
series = df_yearly.set_index("annee")["conso_energie"]
try:
    arima_model = fit_arima(series)
    arima_forecast = forecast_arima(arima_model, steps=3)
    print("ARIMA forecast:\n", arima_forecast)
except Exception as e:
    print("ARIMA fitting failed", e)

# clustering
clustered, kmeans = cluster_collectivities(df_by_collectivite, n_clusters=3)
print("Clusters counts:", clustered["cluster"].value_counts())

# anomalies
anomalies_iso, iso_mod = detect_anomalies_isolation(df_by_collectivite, features=["conso_energie", "depense_energie"], contamination=0.1)
print("Isolation anomalies:", anomalies_iso[anomalies_iso.anomaly].shape[0])

anomalies_z = detect_anomalies_zscore(df_by_collectivite, "ratio")
print("Zscore anomalies:", anomalies_z[anomalies_z.anomaly].shape[0])

from src.monitoring.data_quality import data_quality_report

dq = data_quality_report(df_energy)
print(dq)
