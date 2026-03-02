import sys
from pathlib import Path

# ensure project root is on path when script is executed directly
PROJECT_ROOT = Path(__file__).parents[1].resolve()
sys.path.append(str(PROJECT_ROOT))

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

from src.ingestion.load_data import load_raw_energy_data
from src.preprocessing.clean_data import reshape_energy_data

# additional models and tools
from src.modeling.forecasting import fit_arima, forecast_arima
from src.modeling.clustering import cluster_collectivities
from src.modeling.anomaly import detect_anomalies_isolation, detect_anomalies_zscore
from src.monitoring.data_quality import data_quality_report

import plotly.graph_objects as go

# ============================
#   LOAD & PREPARE DATA
# ============================

df_raw = load_raw_energy_data()
df_energy = reshape_energy_data(df_raw)

df_by_energy = df_energy.groupby("type_energie")[["conso_energie", "depense_energie"]].sum().reset_index()
df_by_collectivite = df_energy.groupby("identifiant")[["conso_energie", "depense_energie"]].sum().reset_index()

# Ratio €/kWh
# évite les divisions par zéro
if "conso_energie" in df_by_collectivite.columns:
    df_by_collectivite["ratio"] = df_by_collectivite["depense_energie"] / df_by_collectivite["conso_energie"].replace(0, pd.NA)
    df_by_collectivite["ratio"] = df_by_collectivite["ratio"].fillna(0)

# Aggregés par année (utilisés pour prévisions)
df_yearly = df_energy.groupby("annee")[["conso_energie", "depense_energie"]].sum().reset_index()

# ============================
#   DASH APP
# ============================

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard Énergie - EPC 2017", style={"textAlign": "center"}),

    dcc.Tabs([

        # ============================
        #   TAB 1 : VUE D’ENSEMBLE
        # ============================
        dcc.Tab(label="Vue d'ensemble", children=[
            html.Div([
                html.Div([
                    html.H3("Consommation totale"),
                    html.P(f"{df_energy['conso_energie'].sum():,.0f} kWh")
                ], className="kpi"),

                html.Div([
                    html.H3("Dépense totale"),
                    html.P(f"{df_energy['depense_energie'].sum():,.0f} €")
                ], className="kpi"),

                html.Div([
                    html.H3("Nombre de collectivités"),
                    html.P(df_energy["identifiant"].nunique())
                ], className="kpi"),

                html.Div([
                    html.H3("Types d'énergie"),
                    html.P(df_energy["type_energie"].nunique())
                ], className="kpi"),
            ], className="kpi-container"),

            dcc.Graph(
                figure=px.bar(df_by_energy, x="type_energie", y="conso_energie",
                              title="Consommation par type d'énergie")
            ),

            dcc.Graph(
                figure=px.line(
                    df_yearly,
                    x="annee",
                    y="conso_energie",
                    title="Évolution annuelle de la consommation"
                )
         ),

            dcc.Graph(
                figure=px.bar(df_by_energy, x="type_energie", y="depense_energie",
                              title="Dépenses par type d'énergie")
            ),
        ]),

        # ============================
        #   TAB 2 : ANALYSE COLLECTIVITÉ
        # ============================
        dcc.Tab(label="Analyse par collectivité", children=[

            html.Label("Choisir une collectivité :"),
            dcc.Dropdown(
                id="collectivite-filter",
                options=[{"label": str(i), "value": i} for i in df_energy["identifiant"].unique()],
                value=None,
                placeholder="Sélectionner une collectivité",
                searchable=True
            ),

            dcc.Graph(id="collectivite-detail"),

            dcc.Graph(
                figure=px.scatter(df_by_collectivite, x="conso_energie", y="depense_energie",
                                  title="Efficacité énergétique (Conso vs Dépense)",
                                  trendline="ols")
            ),
        ]),

        # ============================
        #   TAB 3 : ANALYSE DÉTAILLÉE
        # ============================
        dcc.Tab(label="Analyse détaillée", children=[
            html.Label("Choisir un ou plusieurs types d'énergie :"),
            dcc.Dropdown(
                id="energy-filter",
                options=[{"label": e, "value": e} for e in df_energy["type_energie"].unique()],
                value=["electricite"],
                multi=True
            ),

            dcc.Graph(id="energy-detail-graph")
        ]),

        # ============================
        #   TAB 4 : ANOMALIES
        # ============================
        dcc.Tab(label="Anomalies", children=[
            dcc.Graph(id="anomaly-graph")
        ]),

        # ============================
        #   TAB 5 : MIX ÉNERGÉTIQUE
        # ============================
        dcc.Tab(label="Mix énergétique", children=[
            html.Label("Choisir une collectivité :"),
            dcc.Dropdown(
                id="mix-collectivite",
                options=[{"label": str(i), "value": i} for i in df_energy["identifiant"].unique()],
                value=df_energy["identifiant"].unique()[0]
            ),
            dcc.Graph(id="mix-radar")
        ]),
        # ============================
        #   TAB 6 : PRÉVISIONS
        # ============================
        dcc.Tab(label="Prévisions", children=[
            html.Label("Indicateur :"),
            dcc.Dropdown(
                id="forecast-indicator",
                options=[
                    {"label": "Consommation", "value": "conso_energie"},
                    {"label": "Dépense", "value": "depense_energie"},
                ],
                value="conso_energie",
                clearable=False,
            ),
            html.Br(),
            html.Label("Années à prévoir :"),
            dcc.Slider(
                id="forecast-years",
                min=1,
                max=10,
                step=1,
                value=3,
                marks={i: str(i) for i in range(1, 11)},
            ),
            dcc.Graph(id="forecast-graph")
        ]),
        # ============================
        #   TAB 7 : CLUSTERING
        # ============================
        dcc.Tab(label="Clustering", children=[
            html.Label("Nombre de groupes :"),
            dcc.Slider(
                id="cluster-count",
                min=2,
                max=10,
                step=1,
                value=3,
                marks={i: str(i) for i in range(2, 11)},
            ),
            dcc.Graph(id="cluster-graph")
        ]),
        # ============================
        #   TAB 8 : QUALITÉ DES DONNÉES
        # ============================
        dcc.Tab(label="Qualité des données", children=[
            dcc.Graph(id="dq-table")
        ]),
    ])
])

# ============================
#   CALLBACKS
# ============================

# COLLECTIVITÉ DETAIL
@app.callback(
    Output("collectivite-detail", "figure"),
    Input("collectivite-filter", "value")
)
def update_collectivite(selected_id):
    if selected_id is None:
        return px.bar(title="Sélectionnez une collectivité")

    df_sel = df_energy[df_energy["identifiant"] == selected_id]
    return px.bar(df_sel, x="type_energie", y="conso_energie",
                  title=f"Mix énergétique de la collectivité {selected_id}")


# MULTI-ÉNERGIES
@app.callback(
    Output("energy-detail-graph", "figure"),
    Input("energy-filter", "value")
)
def update_energy_detail(selected_energies):
    df_filtered = df_energy[df_energy["type_energie"].isin(selected_energies)]
    df_grouped = df_filtered.groupby("type_energie")[["conso_energie", "depense_energie"]].sum().reset_index()

    return px.bar(df_grouped, x="type_energie", y="conso_energie",
                  title="Comparaison des énergies sélectionnées")


# ANOMALIES
@app.callback(
    Output("anomaly-graph", "figure"),
    Input("anomaly-graph", "id")
)
def detect_anomalies(_):
    # use isolation forest to highlight anomalies in ratio
    df_ratio = df_by_collectivite.copy()
    df_ratio, _ = detect_anomalies_isolation(df_ratio, features=["ratio"])
    return px.scatter(
        df_ratio,
        x="conso_energie",
        y="ratio",
        color="anomaly",
        title="Anomalies détectées (€/kWh)",
        color_discrete_map={False: "blue", True: "red"},
    )


# PREVISIONS
@app.callback(
    Output("forecast-graph", "figure"),
    Input("forecast-indicator", "value"),
    Input("forecast-years", "value"),
)
def update_forecast(indicator, n_years):
    # build a time series from yearly aggregation
    df = df_yearly.set_index("annee")
    series = df[indicator]

    try:
        model = fit_arima(series)
        forecast_df = forecast_arima(model, steps=n_years)
        # combine historical + forecast
        hist = series.reset_index().rename(columns={indicator: "value"})
        fut = forecast_df[["period", "mean"]].rename(columns={"period": "annee", "mean": "value"})
        df_plot = pd.concat([hist, fut], ignore_index=True)
    except Exception:
        # fallback to simple linear trend
        from src.modeling.simple_trend import fit_linear_trend, predict_future
        model = fit_linear_trend(df.reset_index(), indicator)
        start = df.index.max() + 1
        future_years = list(range(start, start + n_years))
        fut = predict_future(model, future_years).rename(columns={"prediction": "value"})
        hist = df.reset_index().rename(columns={indicator: "value"})
        df_plot = pd.concat([hist, fut], ignore_index=True)

    return px.line(df_plot, x="annee", y="value", title=f"Prévision de {indicator}")


# CLUSTERING
@app.callback(
    Output("cluster-graph", "figure"),
    Input("cluster-count", "value"),
)
def update_clusters(n):
    df_clustered, _ = cluster_collectivities(df_by_collectivite, n_clusters=n)
    return px.scatter(
        df_clustered,
        x="conso_energie",
        y="depense_energie",
        color="cluster",
        title=f"Clusters des collectivités ({n} groupes)",
        hover_data=["identifiant"],
    )


# QUALITÉ DES DONNÉES
@app.callback(
    Output("dq-table", "figure"),
    Input("dq-table", "id"),
)
def show_quality(_):
    dq = data_quality_report(df_energy)
    return go.Figure(
        data=[
            go.Table(
                header=dict(values=list(dq.columns)),
                cells=dict(values=[dq[col] for col in dq.columns]),
            )
        ]
    )


# MIX ÉNERGÉTIQUE (RADAR)
@app.callback(
    Output("mix-radar", "figure"),
    Input("mix-collectivite", "value")
)
def update_radar(selected_id):
    df_sel = df_energy[df_energy["identifiant"] == selected_id]
    df_grouped = df_sel.groupby("type_energie")["conso_energie"].sum().reset_index()

    return px.line_polar(df_grouped, r="conso_energie", theta="type_energie",
                         line_close=True,
                         title=f"Mix énergétique de la collectivité {selected_id}")


if __name__ == "__main__":
    app.run(debug=True)