# Fleet Maintenance Intelligence Hub

A portfolio prototype for fleet maintenance cost analysis, combining CSV preparation, BigQuery analytics, LightGBM/XGBoost models and a local Streamlit interface.

The prediction target is `Maintenance_Cost`. This is a cost estimation experiment, not a validated system for predicting when a vehicle will fail or require maintenance.

## Implemented workflow

| Script | Responsibility | Input and output |
|---|---|---|
| `clean_data.py` | Remove duplicate rows, parse maintenance dates and split records | Source CSV → three local Parquet files |
| `load_to_bigquery.py` | Upload Parquet files using BigQuery load jobs | Local Parquet → three tables in `fleet_warehouse` |
| `analytics_bigquery.py` | Run maintenance cost, sensor and anomaly queries | BigQuery tables → printed results |
| `train_dual_model.py` | Train and evaluate two regressors with five vehicle-grouped folds | Local Parquet → `fleet_dual_model.joblib` |
| `app.py` | Display fleet metrics and run interactive or batch inference | Local Parquet and model artifact → Streamlit interface |

**The training script and dashboard read local files, not BigQuery.** BigQuery analytics and local modelling are separate branches. No scheduler, deployment pipeline or cloud-hosted application is provided by these scripts.

Preparation creates `record_id` from the DataFrame index and extracts a vehicle dimension plus maintenance and telemetry records. The identifier depends on the source ordering; it is not a stable business identifier. Invalid maintenance dates become missing values through `errors="coerce"`.

## BigQuery behaviour

The loader creates a dataset in the EU location and uses `WRITE_TRUNCATE` to replace each destination table. Clustering is configured; time partitioning is not.

Replacing tables avoids appending the same input repeatedly, but the three replacements do not form an atomic publication. A failed load can leave tables from different executions. Existing destination data is overwritten.

The SQL covers maintenance expenditure, sensor averages and global temperature z-scores. Maintenance expenditure alone is not total cost of ownership. The query labelled `Avg_Cost_Per_Vehicle` averages maintenance records rather than first aggregating per vehicle. A date filter in the script does not establish partition pruning, because the loader does not configure partitioning.

## Modelling and validation

- Target transformation: `log1p(Maintenance_Cost)`.
- Features include vehicle attributes, sensor readings, maintenance type and derived load, fuel and usage ratios.
- Five-fold `GroupKFold`, grouped by `Vehicle_ID`, separates vehicles between training and validation within each fold.
- LightGBM uses validation-based early stopping; XGBoost uses a fixed 400 estimators.
- The saved artifact contains the five models from each family, feature names, categorical columns, weights and blend metrics.

Grouping prevents the same vehicle from appearing on both sides of a fold. It does **not** establish that all features were available before a maintenance decision. In particular, the timing of `Maintenance_Type`, `Failure_History` and `Anomalies_Detected` requires investigation. The evaluation is not a chronological forecast backtest.

### Previously reported results

The previous README reported the following five-fold results on 92,000 records. They are retained as historical results, not independently reproduced benchmarks. The source dataset's currency must be verified before treating errors as euros.

| Model | MAE, target units | RMSE, target units | R² |
|---|---:|---:|---:|
| LightGBM | 284.27 | 538.23 | 0.8832 |
| XGBoost | 284.47 | 538.43 | 0.8831 |
| Equal-weight blend | 284.19 | 537.71 | 0.8835 |

**Evaluation and application blending currently differ:** the training script blends the two models' predictions after `expm1`, whereas the application averages fold predictions and blends in log space before `expm1`. The reported blend metrics therefore do not directly evaluate the application's prediction rule.

Other limitations include a fixed reference year of 2024 for vehicle age, no persisted dependency lockfile, no independently held-out final evaluation, and no automated checks for join cardinality or equality of maintenance/telemetry vehicle identifiers. Feature importance and small blend improvements do not establish operational benefit.

## Running the prototype

Run commands from the repository root. Python 3.12 was used in the original setup. Dependencies below are unpinned; this is a starting environment, not a verified reproducible lockfile.

```bash
git clone https://github.com/chouaibmouidis-py/Fleet-Maintenance-Intelligence-Hub.git
cd Fleet-Maintenance-Intelligence-Hub
uv venv --python 3.12
uv pip install pandas numpy pyarrow google-cloud-bigquery google-cloud-bigquery-storage db-dtypes lightgbm xgboost scikit-learn joblib streamlit plotly
```

Provide `logistics_dataset_with_maintenance_required.csv` at the repository root to rebuild the Parquet files. The raw CSV is not included among the tracked files reviewed. The project also contains prepared Parquet files and a serialized model; their presence does not replace source provenance or a reproducible training environment.

Local preparation, training and UI:

```bash
uv run --no-project python clean_data.py
uv run --no-project python train_dual_model.py
uv run --no-project streamlit run app.py
```

For the optional BigQuery branch, configure your own Google Cloud project, enable BigQuery, grant the required permissions and use Application Default Credentials. Replace the hard-coded `PROJECT_ID` in both BigQuery scripts before running them. These operations can incur charges and replace destination tables.

```bash
gcloud auth application-default login
uv run --no-project python load_to_bigquery.py
uv run --no-project python analytics_bigquery.py
```

## Priorities before operational use

1. Establish dataset provenance, licence, units and the intended prediction time.
2. Validate input schemas, keys, joins and feature availability.
3. Use one shared feature and prediction implementation for training and inference.
4. Re-evaluate with a baseline and a held-out period matching the intended use.
5. Pin dependencies, parameterise configuration and add publication safeguards.

## Reuse and data rights

The previous README declared the code public domain under the Unlicense. No standalone licence file was present in the reviewed tree; formalising that declaration remains a documentation task. That declaration does not establish reuse rights for the source dataset, prepared Parquet files or model training data. Dataset provenance and applicable terms must be established separately.
