import mlflow

client = mlflow.tracking.MlflowClient(tracking_uri="mlruns")
exp = client.get_experiment_by_name("iris_classification_ci")
runs = sorted(
    client.search_runs([exp.experiment_id]),
    key=lambda r: r.info.start_time,
    reverse=True,
)
r = runs[0]
print(f"mlruns/{exp.experiment_id}/{r.info.run_id}/artifacts/model")
