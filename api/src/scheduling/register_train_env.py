from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

# Authenticate
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# Register the environment
custom_env = Environment(
    name="training-env",
    description="Training environment for Segformer and UNet",
    conda_file="./env/training-env.yml",  # Ensure correct path
)

ml_client.environments.create_or_update(custom_env)
print("✅ Training environment registered!")