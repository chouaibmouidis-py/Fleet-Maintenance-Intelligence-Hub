from google.cloud import bigquery
import pandas as pd

PROJECT_ID = "wise-aegis-512118-m6"
DATASET_ID = "fleet_warehouse"

client = bigquery.Client(project=PROJECT_ID)

print("=" * 70)
print("ÉTAPE 3 : COUCHE ANALYTIQUE ET BUSINESS INTELLIGENCE (BIGQUERY)")
print("=" * 70)


def run_query(title: str, query: str) -> pd.DataFrame:
  print(f"\n--- {title} ---")
  df = client.query(query).to_dataframe()
  print(df)
  return df


# 1. Analyse financière : les coûts totaux et TCO par modèle de véhicule
q_tco = f"""
SELECT 
    v.Vehicle_Type,
    v.Make_and_Model,
    COUNT(DISTINCT v.Vehicle_ID) AS Total_Vehicles,
    ROUND(SUM(m.Maintenance_Cost), 2) AS Total_Maintenance_Spend,
    ROUND(AVG(m.Maintenance_Cost), 2) AS Avg_Cost_Per_Vehicle
FROM `{PROJECT_ID}.{DATASET_ID}.vehicles` v
JOIN `{PROJECT_ID}.{DATASET_ID}.maintenance` m 
    ON v.Vehicle_ID = m.Vehicle_ID
GROUP BY v.Vehicle_Type, v.Make_and_Model
ORDER BY Total_Maintenance_Spend DESC
LIMIT 5;
"""
run_query("1. Top 5 des modèles les plus coûteux en maintenance (TCO)", q_tco)

# 2. Corrélation Télématique IoT & Pannes (Jointure via record_id)
q_iot_correlation = f"""
SELECT 
    m.Maintenance_Type,
    COUNT(m.record_id) AS Event_Count,
    ROUND(AVG(t.Engine_Temperature), 2) AS Avg_Engine_Temp,
    ROUND(AVG(t.Vibration_Levels), 2) AS Avg_Vibration,
    ROUND(AVG(t.Tire_Pressure), 2) AS Avg_Tire_Pressure,
    ROUND(AVG(t.Fuel_Consumption), 2) AS Avg_Fuel_Consumption
FROM `{PROJECT_ID}.{DATASET_ID}.maintenance` m
JOIN `{PROJECT_ID}.{DATASET_ID}.telemetrie` t 
    ON m.record_id = t.record_id
GROUP BY m.Maintenance_Type
ORDER BY Avg_Engine_Temp DESC;
"""
run_query("2. Analyse IoT : Signature des capteurs par type de panne", q_iot_correlation)

# 3. Détection d'anomalies via Window Functions (Z-Score relatif à la catégorie)
q_anomalies = f"""
WITH FleetStats AS (
    SELECT 
        record_id,
        Vehicle_ID,
        Engine_Temperature,
        Fuel_Consumption,
        AVG(Engine_Temperature) OVER() AS Fleet_Avg_Temp,
        STDDEV(Engine_Temperature) OVER() AS Fleet_Std_Temp,
        AVG(Fuel_Consumption) OVER() AS Fleet_Avg_Fuel,
        STDDEV(Fuel_Consumption) OVER() AS Fleet_Std_Fuel
    FROM `{PROJECT_ID}.{DATASET_ID}.telemetrie`
)
SELECT 
    record_id,
    Vehicle_ID,
    Engine_Temperature,
    ROUND(Fleet_Avg_Temp, 2) AS Fleet_Avg_Temp,
    ROUND((Engine_Temperature - Fleet_Avg_Temp) / NULLIF(Fleet_Std_Temp, 0), 2) AS Temp_ZScore,
    Fuel_Consumption,
    ROUND((Fuel_Consumption - Fleet_Avg_Fuel) / NULLIF(Fleet_Std_Fuel, 0), 2) AS Fuel_ZScore
FROM FleetStats
WHERE (Engine_Temperature - Fleet_Avg_Temp) / NULLIF(Fleet_Std_Temp, 0) > 2.0
LIMIT 5;
"""
run_query("3. Détection d'anomalies thermodynamiques critiques (Window Functions)", q_anomalies)

# 4. Requête partitionnée (Partition Pruning demonstration)
q_partition_pruning = f"""
SELECT 
    DATE(Last_Maintenance_Date) AS Maintenance_Day,
    COUNT(*) AS Operations_Count,
    ROUND(SUM(Maintenance_Cost), 2) AS Daily_Cost
FROM `{PROJECT_ID}.{DATASET_ID}.maintenance`
WHERE Last_Maintenance_Date BETWEEN '2023-01-01' AND '2023-03-31'
GROUP BY Maintenance_Day
ORDER BY Maintenance_Day ASC
LIMIT 5;
"""
run_query("4. Requête optimisée avec Partition Pruning (1er trimestre 2023)", q_partition_pruning)

print("\n" + "=" * 70)
print("✅ TOUTES LES ANALYSES SONT EXÉCUTÉES SUR BIGQUERY AVEC SUCCÈS")
print("=" * 70)
