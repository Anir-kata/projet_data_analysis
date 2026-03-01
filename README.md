# Energy Data Project

## Objectif

Analyser et modéliser les consommations et dépenses énergétiques des collectivités locales françaises (2012–2017), avec un focus sur :
- consommations d'énergie
- dépenses énergétiques
- émissions de CO₂
- secteurs (bâtiments, éclairage public, carburant)
- types d'énergie (électricité, gaz, fioul, ENR, etc.)

## Architecture du projet

- `data/raw/` : données brutes
- `data/processed/` : données nettoyées / agrégées
- `src/` : code Python structuré par étapes du pipeline
  - `ingestion/` : chargement des données brutes
  - `preprocessing/` : nettoyage, standardisation, contrôles qualité
  - `analysis/` : statistiques descriptives, agrégations, corrélations
  - `modeling/` : modèles (régression, ARIMA, clustering, détection d'anomalies)
  - `monitoring/` : rapports de qualité de données
- `notebooks/` : exploration et visualisation
- `dashboard/` : (optionnel) interface de visualisation

## Installation rapide

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Nouvelles fonctionnalités

1. Modèles étendus : ARIMA pour les prévisions, KMeans pour regrouper les collectivités, IsolationForest et z‑score pour détecter des anomalies.
2. Dashboard enrichi : nouveaux onglets "Prévisions", "Clustering" et "Qualité des données" ainsi que des visualisations plus originales.
3. Thème cosy : couleurs pastel/acidulées, arrière‑plans en dégradé et styles CSS modernes.
4. Tests étendus : `test.py` couvre les nouvelles fonctions de modélisation.

Pensez à relancer `pip install -r requirements.txt` après la mise à jour afin d'avoir `statsmodels`.