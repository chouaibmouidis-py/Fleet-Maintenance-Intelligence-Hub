from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Fleet Maintenance Intelligence Hub",
    page_icon="🚛",
    layout="wide",
)

st.title("🚛 Fleet Maintenance Intelligence Hub")
st.markdown(
    "*Système prédictif de maintenance de flotte.* "
)


# 1. Chargement de l'artéfact Dual-Model
@st.cache_resource
def load_dual_model_artifact():
  model_path = Path("fleet_dual_model.joblib")
  if not model_path.exists():
    st.error(
        "Fichier modèle 'fleet_dual_model.joblib' introuvable. Exécute d'abord"
        " train_dual_model.py."
    )
    st.stop()
  return joblib.load(model_path)


artifact = load_dual_model_artifact()
lgb_models = artifact["lgb_models"]
xgb_models = artifact["xgb_models"]
model_features = artifact["features"]
cat_cols = artifact["cat_cols"]
metrics = artifact["metrics"]


# 2. Fonction d'inférence Dual-Model Ensemble (Log-Space Blending)
def predict_ensemble(df_input: pd.DataFrame) -> np.ndarray:
  df_proc = df_input.copy()
  for c in cat_cols:
    if c in df_proc.columns:
      df_proc[c] = df_proc[c].astype("category")

  X_in = df_proc[model_features]

  # Prédictions des 5 folds LightGBM
  preds_lgb = np.mean([m.predict(X_in) for m in lgb_models], axis=0)
  # Prédictions des 5 folds XGBoost
  preds_xgb = np.mean([m.predict(X_in) for m in xgb_models], axis=0)

  # Fusion 50/50 et inversion logarithmique expm1(log)
  blend_log = 0.5 * preds_lgb + 0.5 * preds_xgb
  return np.expm1(blend_log)


# 3. Chargement des données pour le dashboard
@st.cache_data
def load_fleet_data():
  df_v = pd.read_parquet("clean_vehicles.parquet")
  df_m = pd.read_parquet("clean_maintenance.parquet")
  df_t = pd.read_parquet("clean_telemetrie.parquet")
  merged = df_m.merge(df_t, on="record_id")
  if "Vehicle_ID_x" in merged.columns:
    merged["Vehicle_ID"] = merged["Vehicle_ID_x"]
    merged = merged.drop(columns=["Vehicle_ID_x", "Vehicle_ID_y"])
  return merged.merge(df_v, on="Vehicle_ID", how="left")


df_fleet = load_fleet_data()

# Navigation par Onglets
tab1, tab2, tab3 = st.tabs([
    "📊 Tableau de Bord Flotte",
    "🔮 Diagnostic & Prédiction Temps Réel",
    "📂 Inférence & Ingestion par Lot",
])

# ==============================================================================
# ONGLET 1 : DASHBOARD KPI
# ==============================================================================
with tab1:
  st.subheader("Indicateurs opérationnels de la flotte")

  c1, c2, c3, c4 = st.columns(4)
  total_veh = df_fleet["Vehicle_ID"].nunique()
  total_cost = df_fleet["Maintenance_Cost"].sum()
  avg_vibe = df_fleet["Vibration_Levels"].mean()
  perf_r2 = metrics["R2"]

  c1.metric("Véhicules sous Gestion", f"{total_veh:,}")
  c2.metric("Dépenses Maintenance Totales", f"{total_cost:,.2f} €")
  c3.metric("Niveau de Vibration Moyen", f"{avg_vibe:.2f} mm/s")
  c4.metric("Score Modèle Dual-Blend ($R^2$ OOF)", f"{perf_r2:.1%}")

  st.divider()

  col_g1, col_g2 = st.columns(2)

  with col_g1:
    fig_tco = px.histogram(
        df_fleet,
        x="Make_and_Model",
        y="Maintenance_Cost",
        color="Vehicle_Type",
        barmode="group",
        title="Dépenses de Maintenance par Modèle et Type",
        labels={"Maintenance_Cost": "Dépenses (€)", "Make_and_Model": "Modèle"},
    )
    st.plotly_chart(fig_tco, use_container_width=True)

  with col_g2:
    fig_scatter = px.scatter(
        df_fleet.sample(min(3000, len(df_fleet))),
        x="Vibration_Levels",
        y="Maintenance_Cost",
        color="Maintenance_Type",
        title="Impact des Vibrations sur les Coûts de Réparation (Échantillon)",
        labels={
            "Vibration_Levels": "Vibrations (mm/s)",
            "Maintenance_Cost": "Coût (€)",
        },
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================================================================
# ONGLET 2 : DIAGNOSTIC & PRÉDICTION EN DIRECT (SIMULATEUR)
# ==============================================================================
with tab2:
  st.subheader("Simulateur d'incidents et estimation financière")
  st.write(
      "Ajustez les paramètres d'un véhicule pour estimer en temps réel le coût"
      " d'intervention via l'Ensemble Dual-Model :"
  )

  col_in1, col_in2, col_in3 = st.columns(3)

  with col_in1:
    v_type = st.selectbox(
        "Type de Véhicule", df_fleet["Vehicle_Type"].unique()
    )
    v_model = st.selectbox(
        "Modèle",
        df_fleet[df_fleet["Vehicle_Type"] == v_type]["Make_and_Model"].unique(),
    )
    m_type = st.selectbox(
        "Type d'Intervention Prévue", df_fleet["Maintenance_Type"].unique()
    )
    route = st.selectbox("Type de Trajet", df_fleet["Route_Info"].unique())

  with col_in2:
    year = st.slider("Année de Mise en Circulation", 2010, 2024, 2018)
    hours = st.slider("Heures d'Usage Cumulées", 1, 10000, 2500)
    actual_load = st.number_input("Charge Embarquée (kg)", 100.0, 50000.0, 3200.0)
    load_capacity = st.number_input("Capacité Max (kg)", 500.0, 50000.0, 3500.0)

  with col_in3:
    vibration = st.slider(
        "Niveau de Vibration (mm/s)", 0.0, 10.0, float(df_fleet["Vibration_Levels"].median())
    )
    oil_quality = st.slider(
        "Qualité de l'Huile (%)", 0.0, 100.0, float(df_fleet["Oil_Quality"].median())
    )
    tire_pressure = st.slider(
        "Pression des Pneus (PSI)", 20.0, 45.0, float(df_fleet["Tire_Pressure"].median())
    )
    fuel_cons = st.slider(
        "Consommation (L/100km)", 5.0, 25.0, float(df_fleet["Fuel_Consumption"].median())
    )

  if st.button("🚀 Calculer la prévision de maintenance", type="primary"):
    # Reconstruction de la ligne d'inférence avec Feature Engineering strict
    sample = pd.DataFrame([{
        "Vehicle_Type": v_type,
        "Make_and_Model": v_model,
        "Year_of_Manufacture": year,
        "Usage_Hours": hours,
        "Route_Info": route,
        "Load_Capacity": load_capacity,
        "Actual_Load": actual_load,
        "Maintenance_Type": m_type,
        "Engine_Temperature": 120.0,
        "Tire_Pressure": tire_pressure,
        "Fuel_Consumption": fuel_cons,
        "Battery_Status": 85.0,
        "Vibration_Levels": vibration,
        "Oil_Quality": oil_quality,
        "Failure_History": 1,
        "Anomalies_Detected": 0,
    }])

    # Features physiques
    sample["Load_Ratio"] = sample["Actual_Load"] / (
        sample["Load_Capacity"] + 1e-5
    )
    sample["Is_Overloaded"] = (sample["Load_Ratio"] > 1.0).astype(int)
    sample["Fuel_Per_Ton"] = sample["Fuel_Consumption"] / (
        sample["Actual_Load"] + 1.0
    )
    sample["Vehicle_Age"] = 2024 - sample["Year_of_Manufacture"]
    sample["Usage_Intensity"] = sample["Usage_Hours"] / (
        sample["Vehicle_Age"] + 1.0
    )

    # Inférence avec le Blend
    prediction = predict_ensemble(sample)[0]

    st.markdown("### 📋 Résultat du Diagnostic")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
      st.metric(
          "Coût d'intervention prévu (Dual-Model)", f"{prediction:,.2f} €"
      )

    with res_col2:
      if prediction > 2000 or sample["Load_Ratio"].iloc[0] > 1.0:
        st.error(
            "⚠️ RISQUE CRITIQUE : Intervention immédiate recommandée (Usure"
            " anormale ou surcharge détectée)."
        )
      elif prediction > 1000:
        st.warning(
            "⚡ ATTENTION : Maintenance préventive à planifier sous 15 jours."
        )
      else:
        st.success(
            "✅ ÉTAT NOMINAL : Paramètres mécaniques dans la plage standard."
        )

# ==============================================================================
# ONGLET 3 : INFÉRENCE PAR LOT (BATCH)
# ==============================================================================
with tab3:
  st.subheader("Prédiction et injection de données par lot (Batch)")
  st.write(
      "Déposez un fichier CSV contenant des relevés pour générer les prévisions"
      " en masse :"
  )

  uploaded_file = st.file_uploader(
      "Sélectionner un fichier CSV de relevés", type=["csv"]
  )

  if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file)
    st.write(f"📂 **{len(df_new)}** enregistrements chargés.")

    if st.button("Lancer les Prédictions sur le Lot"):
      df_proc = df_new.copy()
      df_proc["Load_Ratio"] = df_proc["Actual_Load"] / (
          df_proc["Load_Capacity"] + 1e-5
      )
      df_proc["Is_Overloaded"] = (df_proc["Load_Ratio"] > 1.0).astype(int)
      df_proc["Fuel_Per_Ton"] = df_proc["Fuel_Consumption"] / (
          df_proc["Actual_Load"] + 1.0
      )
      df_proc["Vehicle_Age"] = 2024 - df_proc["Year_of_Manufacture"]
      df_proc["Usage_Intensity"] = df_proc["Usage_Hours"] / (
          df_proc["Vehicle_Age"] + 1.0
      )

      # Prédictions de l'ensemble
      df_proc["Predicted_Cost"] = predict_ensemble(df_proc)

      st.dataframe(
          df_proc[
              ["Vehicle_ID", "Make_and_Model", "Usage_Hours", "Predicted_Cost"]
          ].head(20)
      )

      fig_dist = px.box(
          df_proc,
          x="Make_and_Model",
          y="Predicted_Cost",
          color="Make_and_Model",
          title="Distribution des Coûts Prévus par Modèle",
      )
      st.plotly_chart(fig_dist, use_container_width=True)
      st.success("✅ Inférence par lot terminée avec succès !")