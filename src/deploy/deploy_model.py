from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import uuid

# Authenticate to Azure ML
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# Unique endpoint name
endpoint_name = f"best-model-endpoint-{uuid.uuid4().hex[:8]}"

# Register model from Azure Blob Storage
model = Model(
    path="azureml://datastores/workspaceblobstore/paths/models/best_model.pth",  # Update with your Blob path
    name="best_model",
    description="Trained deep learning model",
    type=AssetTypes.CUSTOM_MODEL,
)

ml_client.models.create_or_update(model)

# Define the endpoint
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="Endpoint for best_model",
    auth_mode="key"  # Can also use 'aml_token' for authentication
)

ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Define the deployment
deployment = ManagedOnlineDeployment(
    name="best-model-deployment",
    endpoint_name=endpoint_name,
    model=model.id,
    instance_type="Standard_NC6",  # GPU VM for inference
    instance_count=1,
    environment="azureml:inference-env:latest",  # Reference the environment you created
    code_configuration={"code": "./src/deploy", "scoring_script": "score.py"},
)

ml_client.online_deployments.begin_create_or_update(deployment).result()

# Print endpoint details
print(f"Deployment successful! Endpoint URL: {ml_client.online_endpoints.get(endpoint_name).scoring_uri}")
