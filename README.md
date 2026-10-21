# 🚛 Fleet Maintenance Intelligence Hub

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Google BigQuery](https://img.shields.io/badge/cloud-Google%20BigQuery-4285F4.svg)](https://cloud.google.com/bigquery)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: Free](https://img.shields.io/badge/licence-Libre%20de%20droits-green.svg)](#-licence--conditions-dutilisation)

```Plateforme Cloud-Native d'analyse télématique et de maintenance prédictive pour flottes logistiques.  
Pipeline ETL vers Google BigQuery, analyses SQL distribuées et architecture Dual-Model (LightGBM + XGBoost) validée par GroupKFold sans fuite de données.```

---

## 🏗️ Architecture du Système

Le projet adopte les principes de l'Architecture Medallion (séparation stockage, calcul distribué et inférence) :

* **Données Brutes** : Relevés IoT et historique d'interventions (CSV).
* **Couche Silver (`clean_data.py`)** : Normalisation, typage strict des dates et export Apache Parquet (compression Snappy).
* **Couche Gold (`load_to_bigquery.py`)** : Ingestion BigQuery asynchrone en mémoire constante O(1), tables clusterisées par Vehicle_ID et Maintenance_Type.
* **Analytique SQL (`analytics_bigquery.py`)** : Calcul du TCO, signatures de capteurs et détection d'anomalies par Window Functions.
* **Machine Learning (`train_dual_model.py`)** : Validation 5-Fold GroupKFold étanche sur Vehicle_ID, cible log1p, ensemble Dual-Model (LightGBM + XGBoost).
* **Application Web (`app.py`)** : Dashboard opérationnel, simulateur d'incident et inférence par lot sous Streamlit.

[Données IoT CSV] ──(clean_data.py)──► [Parquet Snappy] ──(load_to_bigquery.py)──► [BigQuery Gold]
                                                                                      │
[Dashboard Streamlit] ◄──(app.py)── [Dual-Model Ensemble] ◄──(train_dual_model.py)────┘

---

## ⚡ Points Clés d'Ingénierie

### 1. Data Engineering & Cloud (Google BigQuery)
* **Ingestion en Mémoire Constante O(1)** : streaming direct des fichiers Parquet vers l'API BigQuery sans chargement lourd en mémoire vive.
* **Idempotence & Résilience** : création automatique du dataset en région UE (`exists_ok=True`) et réécriture contrôlée (`WRITE_TRUNCATE`) pour interdire les doublons.
* **Stockage Optimisé (Clustering)** : indexation physique des tables selon Vehicle_ID et Maintenance_Type pour accélérer les requêtes d'analyse.

### 2. Feature Engineering Physique
* **`Load_Ratio = Actual_Load / Load_Capacity`** : surveillance des surcharges mécaniques destructrices.
* **`Usage_Intensity = Usage_Hours / (Vehicle_Age + 1)`** : détection de l'usure anormale par rapport à l'âge du camion.
* **`Fuel_Per_Ton = Fuel_Consumption / (Actual_Load + 1)`** : rendement énergétique réel par tonne transportée.
* *Validation expérimentale* : `Load_Ratio` et `Usage_Intensity` figurent directement dans le **Top 5 des variables les plus importantes** du modèle.

### 3. Machine Learning (Standards Kaggle)
* **Zéro Fuite de Données (Data Leakage)** : découpage 5-Fold GroupKFold groupé strictement par Vehicle_ID. Aucun camion du jeu de test n'apparaît dans le jeu d'entraînement.
* **Stabilisation Logarithmique** : cible entraînée sur `log1p(Maintenance_Cost)` pour modérer l'impact des pannes extrêmes, puis inversée via `expm1()`.
* **Ensembling Dual-Model** : combinaison pondérée 50/50 des prédictions de LightGBM (division par feuilles) et XGBoost (arbres par histogrammes avec support catégoriel natif).

---

## 📊 Benchmark Comparatif des Modèles (Out-Of-Fold)

Résultats évalués sur 92 000 enregistrements en validation croisée 5-Fold (en euros réels) :

| Modèle                            | MAE (€)      | RMSE (€)     | R² Score   |
| :---                              | :---:        | :---:        | :---:      |
| LightGBM Seul                     | 284.27 €     | 538.23 €     | 0.8832     |
| XGBoost Seul                      | 284.47 €     | 538.43 €     | 0.8831     |
| **DUAL-MODEL BLEND (LGBM + XGB)** | **284.19 €** | **537.71 €** | **0.8835** |

*Le modèle combiné surpasse les deux algorithmes individuels sur l'ensemble des métriques d'erreur.*

---

## 🚀 Installation & Reproduction

Ce projet s'exécute avec le gestionnaire d'environnement `uv` sous Python 3.12.

### 1. Préparation de l'environnement
```bash
git clone https://github.com/ton-profil/fleet_analytics.git
cd fleet_analytics

# Création et activation de l'environnement virtuel
uv venv --python 3.12
source .venv/bin/activate

# Installation des dépendances
uv pip install google-cloud-bigquery google-cloud-bigquery-storage pyarrow db-dtypes "cryptography<49" lightgbm xgboost scikit-learn joblib streamlit plotly
```

### 2. Authentification Google Cloud (ADC)
```bash
gcloud auth application-default login
gcloud config set project wise-aegis-512118-m6
```

### 3. Exécution de la chaîne complète
```bash

# 1. Nettoyage et génération des tables Parquet
uv run python clean_data.py

# 2. Ingestion Cloud vers Google BigQuery
uv run python load_to_bigquery.py

# 3. Calcul des métriques SQL et Z-Scores
uv run python analytics_bigquery.py

# 4. Entraînement du Dual-Model Ensemble
uv run python train_dual_model.py

# 5. Démarrage de l'interface Streamlit
uv run streamlit run app.py
```

---

## 🔮 Roadmap : Version 2 (v2)

La version actuelle (v1) constitue un socle complet et robuste. La version 2 (v2) se concentrera sur l'optimisation avancée des performances :

- [ ] **Optimisation Bayésienne avec Optuna** :
  * Recherche automatisée des hyperparamètres clés (`learning_rate`, `num_leaves`, `max_depth`, régularisations L1/L2).
  * Optimisation dynamique du coefficient de fusion alpha entre LightGBM et XGBoost.
- [ ] **Intégration d'un 3e Modèle (CatBoost)** :
  * Intégration d'arbres symétriques (Oblivious Trees) et comparaison avec une couche de Stacking Ridge.
- [ ] **Déploiement Conteneurisé (Docker + Cloud Run)** :
  * Création d'une image Docker multi-stage pour héberger l'application Streamlit sur Google Cloud Run avec pipeline CI/CD GitHub Actions.
- [ ] **Inférence Distribuée BigQuery ML** :
  * Export des artefacts de modèle pour effectuer les prédictions directement en SQL au sein de BigQuery.

---

## 📄 Licence & Conditions d'Utilisation

**Ce projet est entièrement libre de droits et ouvert (Domaine Public / Unlicense).**  
**Aucune licence n'est requise** : vous êtes totalement libre de copier, modifier, redistribuer ou intégrer ce code dans vos projets académiques, personnels ou commerciaux, sans restriction ni contrainte légale.

---

**Auteur** : Chouaib MOUID  
*Ingénieur Data & Analytics*
