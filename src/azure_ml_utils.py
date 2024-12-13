from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential



def get_ml_client():
    """
    Returns an Azure MLClient instance for interacting with the Azure ML workspace.

    Returns:
        WS: Initialized Azure Workspace
        MLClient: Configured MLClient instance.
    """

    WS = Workspace.from_config()
    credential = DefaultAzureCredential()
    return MLClient(credential, WS.subscription_id, WS.resource_group, WS.name), WS