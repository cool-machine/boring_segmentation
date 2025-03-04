from azure.ai.ml import MLClient
from azure.ai.ml.entities import Command
from azure.identity import DefaultAzureCredential

# Authenticate
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# Define the training job
train_segformer_job = Command(
    code="./src/training",  # Path to training scripts
    command="python train_segformer.py --epochs 20 --lr 0.0005",
    environment="azureml:training-env:latest",
    compute="gpu-cluster",
    display_name="scheduled-segformer-training",
)

# Submit the training job
job = ml_client.jobs.create_or_update(train_segformer_job)
print(f"🚀 Training job scheduled! Job name: {job.name}")
