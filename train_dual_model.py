from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
import xgboost as xgb

print("=" * 75)
print("🏆 ARCHITECTURE DUAL-MODEL: LIGHTGBM + XGBOOST (GROUP-KFOLD)")
print("=" * 75)

# 1. Chargement des données Parquet
print("📂 Chargement des données Parquet Silver...")
df_v = pd.read_parquet("clean_vehicles.parquet")
df_t = pd.read_parquet("clean_telemetrie.parquet")
df_m = pd.read_parquet("clean_maintenance.parquet")

# 2. Jointure Fact-to-Fact sur record_id + Dimension sur Vehicle_ID
print("🔗 Fusion des données...")
df = df_m.merge(df_t, on="record_id")
if "Vehicle_ID_x" in df.columns:
  df["Vehicle_ID"] = df["Vehicle_ID_x"]
  df = df.drop(columns=["Vehicle_ID_x", "Vehicle_ID_y"])

df = df.merge(df_v, on="Vehicle_ID", how="left")

# 3. FEATURE ENGINEERING
print("⚙️ Ingénierie des variables physiques...")
df["Load_Ratio"] = df["Actual_Load"] / (df["Load_Capacity"] + 1e-5)
df["Is_Overloaded"] = (df["Load_Ratio"] > 1.0).astype(int)
df["Fuel_Per_Ton"] = df["Fuel_Consumption"] / (df["Actual_Load"] + 1.0)
current_year = 2024
df["Vehicle_Age"] = current_year - df["Year_of_Manufacture"]
df["Usage_Intensity"] = df["Usage_Hours"] / (df["Vehicle_Age"] + 1.0)

# 4. Préparation des variables et transformation log de la target
target = "Maintenance_Cost"
ignore_cols = [target, "record_id", "Vehicle_ID", "Last_Maintenance_Date"]
features = [col for col in df.columns if col not in ignore_cols]

# Typage catégoriel natif (supporté par LightGBM et XGBoost)
cat_cols = ["Maintenance_Type", "Vehicle_Type", "Make_and_Model", "Route_Info"]
for c in cat_cols:
  if c in df.columns:
    df[c] = df[c].astype("category")

X = df[features]
y = df[target]
groups = df["Vehicle_ID"]

# Transformation log(1 + y) pour neutraliser les queues épaisses
y_log = np.log1p(y)

print(f"📊 Dataset : {X.shape[0]} lignes | {X.shape[1]} features | Cible log1p")

# 5. Validation Croisée GroupKFold (5 Folds étanches sur Vehicle_ID)
N_SPLITS = 5
gkf = GroupKFold(n_splits=N_SPLITS)

oof_lgb = np.zeros(len(df))
oof_xgb = np.zeros(len(df))
lgb_models = []
xgb_models = []

print(f"\n🔄 Lancement du 5-Fold GroupKFold...")

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
  print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")

  X_tr, y_tr = X.iloc[train_idx], y_log.iloc[train_idx]
  X_va, y_va = X.iloc[val_idx], y_log.iloc[val_idx]

  # --- MODÈLE 1 : LightGBM ---
  model_lgb = lgb.LGBMRegressor(
      n_estimators=400,
      learning_rate=0.03,
      num_leaves=31,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42 + fold,
      n_jobs=-1,
  )
  model_lgb.fit(
      X_tr,
      y_tr,
      eval_set=[(X_va, y_va)],
      callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
  )
  oof_lgb[val_idx] = model_lgb.predict(X_va)
  lgb_models.append(model_lgb)

  # --- MODÈLE 2 : XGBoost ---
  model_xgb = xgb.XGBRegressor(
      n_estimators=400,
      learning_rate=0.03,
      max_depth=6,
      subsample=0.8,
      colsample_bytree=0.8,
      tree_method="hist",
      enable_categorical=True,
      random_state=42 + fold,
      n_jobs=-1,
  )
  model_xgb.fit(
      X_tr,
      y_tr,
      eval_set=[(X_va, y_va)],
      verbose=False,
  )
  oof_xgb[val_idx] = model_xgb.predict(X_va)
  xgb_models.append(model_xgb)

  # Métrique intermédiaire fold en euros réels
  fold_lgb_mae = mean_absolute_error(
      np.expm1(y_va), np.expm1(oof_lgb[val_idx])
  )
  fold_xgb_mae = mean_absolute_error(
      np.expm1(y_va), np.expm1(oof_xgb[val_idx])
  )
  print(f"  • Fold {fold + 1} MAE : LGBM = {fold_lgb_mae:.2f} € | XGBoost = {fold_xgb_mae:.2f} €")

# 6. Évaluation globale Out-Of-Fold (OOF) en Euros Réels
# Rétablissement de l'échelle réelle : expm1(log)
y_true = df[target].values
preds_lgb = np.expm1(oof_lgb)
preds_xgb = np.expm1(oof_xgb)

# Ensembling : Blend 50/50
preds_blend = 0.5 * preds_lgb + 0.5 * preds_xgb


def get_metrics(y_real, y_pred):
  return {
      "MAE": mean_absolute_error(y_real, y_pred),
      "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
      "R2": r2_score(y_real, y_pred),
  }


m_lgb = get_metrics(y_true, preds_lgb)
m_xgb = get_metrics(y_true, preds_xgb)
m_blend = get_metrics(y_true, preds_blend)

print("\n" + "=" * 75)
print("📊 BENCHMARK COMPARATIF SUR LES PRÉDICTIONS OUT-OF-FOLD (OOF)")
print("=" * 75)
summary_df = pd.DataFrame([
    {
        "Modèle": "LightGBM Seul",
        "MAE (€)": f"{m_lgb['MAE']:.2f}",
        "RMSE (€)": f"{m_lgb['RMSE']:.2f}",
        "R² Score": f"{m_lgb['R2']:.4f}",
    },
    {
        "Modèle": "XGBoost Seul",
        "MAE (€)": f"{m_xgb['MAE']:.2f}",
        "RMSE (€)": f"{m_xgb['RMSE']:.2f}",
        "R² Score": f"{m_xgb['R2']:.4f}",
    },
    {
        "Modèle": "🔥 DUAL-MODEL BLEND (LGBM + XGB)",
        "MAE (€)": f"{m_blend['MAE']:.2f}",
        "RMSE (€)": f"{m_blend['RMSE']:.2f}",
        "R² Score": f"{m_blend['R2']:.4f}",
    },
])
print(summary_df.to_string(index=False))
print("=" * 75)

# 7. Sauvegarde du pipeline Dual-Model pour la production
artifact = {
    "lgb_models": lgb_models,
    "xgb_models": xgb_models,
    "blend_weights": (0.5, 0.5),
    "features": features,
    "cat_cols": cat_cols,
    "metrics": m_blend,
}
joblib.dump(artifact, "fleet_dual_model.joblib")
print(
    "\n💾 Pipeline Dual-Model (10 modèles K-Fold) sauvegardé dans"
    " 'fleet_dual_model.joblib' !"
)
