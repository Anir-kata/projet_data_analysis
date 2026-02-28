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
  - `modeling/` : modèles simples de tendance (régression linéaire)
  - `monitoring/` : rapports de qualité de données
- `notebooks/` : exploration et visualisation
- `dashboard/` : (optionnel) interface de visualisation

## Installation rapide

```bash
python -m venv .venv
source .venv/bin/activate  # ou équivalent Windows
pip install -r requirements.txt