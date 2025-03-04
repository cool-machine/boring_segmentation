import os
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, CommandComponent, Command
from azure.identity import DefaultAzureCredential

# ----------------------------------------------------------------
# Step 1: Set the working directory to the project root.
ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)
print(f"✅ Working directory set to: {os.getcwd()}")

# ----------------------------------------------------------------
# Step 2: Locate the training environment YAML file.
env_file_path = ROOT_DIR / "env" / "training-env.yml"
if not env_file_path.exists():
    print("🚨 ERROR: training-env.yml not found!")
    os.system(f"ls -la {ROOT_DIR / 'env'}")  # Debug: list contents of env folder
    raise FileNotFoundError(f"🚨 ERROR: training-env.yml not found at {env_file_path}")
print(f"✅ Found training environment file at: {env_file_path}")

# ----------------------------------------------------------------
# Step 3: Create the MLClient from config.
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# ----------------------------------------------------------------
# Step 4: Define and register the environment with a Docker base image.
# We now use the 'image' parameter to specify the Docker image.
training_env = Environment(
    name="training-env-inline",
    description="Training environment for Segformer and UNet",
    conda_file=str(env_file_path),
    image="mcr.microsoft.com/azureml/base:latest"
)
# Register (or update) the environment.
training_env = ml_client.environments.create_or_update(training_env)
print(f"✅ Environment registered: {training_env.name}")

# ----------------------------------------------------------------
# Step 5: Specify the existing compute instance name.
compute_name = "gpu-c24c-r220rm64m-c415c"  # Replace with your actual compute instance name

# ----------------------------------------------------------------
# Step 6: Define a CommandComponent for the training job.
train_component = CommandComponent(
    name="train_segformer_component",
    display_name="Segformer Training Component",
    description="Component to train the Segformer model",
    command="python train_segformer.py --epochs 20 --lr 0.0005",
    environment=training_env,
    code=str(ROOT_DIR / "src" / "training"),
)

# ----------------------------------------------------------------
# Step 7: Create and submit the training job.
# Note: We explicitly pass 'environment=training_env' at the job level.
train_job = Command(
    component=train_component,
    environment=training_env,
    compute=compute_name,
    display_name="on-demand-segformer-training",
    experiment_name="img-segmentation-exp",
)

job_response = ml_client.jobs.create_or_update(train_job)
print(f"🚀 Training job submitted successfully! Job ID: {job_response.name}")
