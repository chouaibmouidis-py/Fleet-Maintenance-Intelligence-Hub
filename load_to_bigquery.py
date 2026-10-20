from pathlib import Path
from google.cloud import bigquery

PROJECT_ID = "wise-aegis-512118-m6"
DATASET_ID = "fleet_warehouse"

print("🔌 Connexion à Google BigQuery (ADC)...")
client = bigquery.Client(project=PROJECT_ID)

# Garantir l'existence du dataset
dataset_ref = bigquery.DatasetReference(PROJECT_ID, DATASET_ID)
dataset = bigquery.Dataset(dataset_ref)
dataset.location = "EU"
client.create_dataset(dataset, exists_ok=True)
print(f"📦 Dataset '{DATASET_ID}' validé sur Google Cloud.")

# Définition des configurations avancées par table
# Chaque table a sa propre stratégie d'optimisation (Partitioning / Clustering)
tables_pipeline = {
    "vehicles": {
        "file": "clean_vehicles.parquet",
        "config": bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            clustering_fields=["Vehicle_Type", "Vehicle_ID"],
        ),
    },
    "telemetrie": {
        "file": "clean_telemetrie.parquet",
        "config": bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            clustering_fields=["Vehicle_ID"],
        ),
    },
    "maintenance": {
        "file": "clean_maintenance.parquet",
        "config": bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            # Clustering pur : gratuit, sans aucune expiration de date
            clustering_fields=["Vehicle_ID", "Maintenance_Type"],
        ),
    },
}

active_jobs = {}

# 1. Déclenchement asynchrone des flux Parquet
for table_name, meta in tables_pipeline.items():
  file_path = Path(meta["file"])
  if not file_path.exists():
    raise FileNotFoundError(f"Fichier manquant : {file_path}")

  table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"
  print(f"Ingestion optimisée (Parquet) pour '{table_name}'...")

  with open(file_path, "rb") as source_file:
    load_job = client.load_table_from_file(
        source_file, table_id, job_config=meta["config"]
    )
    active_jobs[table_name] = load_job

# 2. Barrière de synchronisation parallèle
print("\n⏳ Traitement et indexation en cours sur BigQuery...")
for table_name, job in active_jobs.items():
  job.result()
  print(
      f"✅ Table '{table_name}' ingérée, partitionnée et clusterisée"
      f" ({job.output_rows} lignes)."
  )

print(
    "\n💉 Ingestion BigQuery aux standards GCP Data Engineering terminée avec"
    " succès !"
)
