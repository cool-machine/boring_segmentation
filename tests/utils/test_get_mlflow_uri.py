import os
from unittest.mock import patch, MagicMock
from src.utils.helpers import get_mlflow_uri

@patch.dict(os.environ, {
    "AZURE_SUBSCRIPTION_ID": "test-sub-id",
    "AZURE_RESOURCE_GROUP": "test-resource-group",
    "AZURE_WORKSPACE_NAME": "test-workspace-name"
})

# with patch("src.utils.config.AML_PROPERTIES", {
#     "subscription_id": "test-sub-id",
#     "resource_group": "test-resource-group",
#     "workspace_name": "test-workspace-name"
# }):
#     from src.utils.helpers import get_mlflow_uri


@patch("src.utils.helpers.MLClient")
@patch("src.utils.helpers.DefaultAzureCredential")
def test_get_mlflow_uri(mock_credential, mock_mlclient):
    # Mock Azure Credential
    mock_credential.return_value = MagicMock()

    # Mock MLClient and Workspace response
    mock_client_instance = MagicMock()
    mock_workspace = MagicMock()
    mock_workspace.mlflow_tracking_uri = "https://test-mlflow-uri"
    mock_client_instance.workspaces.get.return_value = mock_workspace
    mock_mlclient.return_value = mock_client_instance

    # Call the function
    mlflow_uri = get_mlflow_uri()

    # Assertions
    assert mlflow_uri == "https://test-mlflow-uri"
    mock_mlclient.assert_called_once_with(
        mock_credential.return_value,
        subscription_id="test-sub-id",
        resource_group_name="test-resource-group",
        workspace_name="test-workspace-name"
    )
    
    mock_client_instance.workspaces.get.assert_called_once_with("test-workspace-name")