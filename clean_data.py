from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("logistics_dataset_with_maintenance_required.csv")

if not RAW_DATA_PATH.exists():
  raise FileNotFoundError(f"Le fichier brut {RAW_DATA_PATH} est introuvable.")

print("📂 Chargement du dataset brut...")
df = pd.read_csv(RAW_DATA_PATH)
df = df.drop_duplicates()

# 1. Clé technique de synchronisation (Surrogate Key)
# Permet de relier la télémétrie et la maintenance ligne par ligne sans produit cartésien
df["record_id"] = df.index.astype("int64")

# 2. Typage strict de la date
if "Last_Maintenance_Date" in df.columns:
  df["Last_Maintenance_Date"] = pd.to_datetime(
      df["Last_Maintenance_Date"], errors="coerce"
  )

# 3. Table Dimensions : Véhicules (statique et dédupliquée)
df_vehicles = (
    df[[
        "Vehicle_ID",
        "Make_and_Model",
        "Year_of_Manufacture",
        "Vehicle_Type",
        "Load_Capacity",
    ]]
    .drop_duplicates(subset=["Vehicle_ID"])
    .copy()
)

# 4. Table Faits IoT : Télémétrie
cols_telemetrie = [
    "record_id",
    "Vehicle_ID",
    "Usage_Hours",
    "Route_Info",
    "Actual_Load",
    "Engine_Temperature",
    "Tire_Pressure",
    "Fuel_Consumption",
    "Battery_Status",
    "Vibration_Levels",
    "Oil_Quality",
]
df_telemetrie = df[cols_telemetrie].copy()

# 5. Table Faits Opérationnels : Maintenance
cols_maintenance = [
    "record_id",
    "Vehicle_ID",
    "Last_Maintenance_Date",
    "Maintenance_Type",
    "Maintenance_Cost",
    "Failure_History",
    "Anomalies_Detected",
]
df_maintenance = df[cols_maintenance].copy()

# 6. Export au format Parquet (optimisé pour BigQuery, typage préservé)
print("💾 Sérialisation en format Apache Parquet (compression Snappy)...")
df_vehicles.to_parquet("clean_vehicles.parquet", index=False)
df_telemetrie.to_parquet("clean_telemetrie.parquet", index=False)
df_maintenance.to_parquet("clean_maintenance.parquet", index=False)

print(
    f"✅ Nettoyage terminé : {len(df_vehicles)} véhicules, {len(df_telemetrie)} relevés synchronisés."
)