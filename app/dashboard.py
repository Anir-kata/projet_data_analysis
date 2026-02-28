import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

from src.ingestion.load_data import load_raw_energy_data
from src.preprocessing.clean_data import reshape_energy_data

# ============================
#   LOAD & PREPARE DATA
# ============================

df_raw = load_raw_energy_data()
df_energy = reshape_energy_data(df_raw)

df_by_energy = df_energy.groupby("type_energie")[["conso_energie", "depense_energie"]].sum().reset_index()
df_by_collectivite = df_energy.groupby("identifiant")[["conso_energie", "depense_energie"]].sum().reset_index()

# Ratio €/kWh
df_by_collectivite["ratio"] = df_by_collectivite["depense_energie"] / df_by_collectivite["conso_energie"]

# ============================
#   DASH APP
# ============================

app = dash.Dash(__name__)

app.layout = html.Div(id="app-container", children=[
    html.H1("Dashboard Énergie - EPC 2017", style={"textAlign": "center"}),

    # MODE SOMBRE
    dcc.Checklist(
        id="theme-switch",
        options=[{"label": "Mode sombre", "value": "dark"}],
        value=[],
        style={"margin": "20px"}
    ),

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
    ])
])

# ============================
#   CALLBACKS
# ============================

# MODE SOMBRE
@app.callback(
    Output("app-container", "className"),
    Input("theme-switch", "value")
)
def toggle_theme(values):
    return "dark-theme" if "dark" in values else "light-theme"


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
    df_ratio = df_by_collectivite.copy()

    Q1 = df_ratio["ratio"].quantile(0.25)
    Q3 = df_ratio["ratio"].quantile(0.75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR

    df_ratio["anomaly"] = df_ratio["ratio"] > threshold

    return px.scatter(df_ratio, x="conso_energie", y="ratio",
                      color="anomaly",
                      title="Anomalies (€/kWh)")


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