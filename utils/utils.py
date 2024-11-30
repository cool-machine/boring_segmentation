import keras 
from keras.optimizers import Adam
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential



def keras_optimizer():
    return keras.optimizers.Adam(learning_rate=0.0001)


def get_mlflow_uri():
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    return mlflow_tracking_uri