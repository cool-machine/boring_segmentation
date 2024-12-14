from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from src.utils.config import AML_PROPERTIES



SUB_ID = AML_PROPERTIES["subscription_id"]
RSRC_GR = AML_PROPERTIES["resource_group"]
WRKSP_NM = AML_PROPERTIES["workspace_name"]

# Authenticate and connect to Azure ML Workspace
def get_mlflow_uri():
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id=SUB_ID, resource_group_name=RSRC_GR, workspace_name=WRKSP_NM)

    # Retrieve MLflow tracking URI
    mlflow_tracking_uri = ml_client.workspaces.get(WRKSP_NM).mlflow_tracking_uri
    print(f"MLflow Tracking URI: {mlflow_tracking_uri}")

    return mlflow_tracking_uri

