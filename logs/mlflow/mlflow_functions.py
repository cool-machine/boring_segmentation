from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def get_mlflow_uri():
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    return mlflow_tracking_uri

def set_tracking_uri():
    mlflow_tracking_uri = get_mlflow_uri()
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    