import os
import pytest
import tensorflow as tf
import sys

# Add the src directory to Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.utils import helpers

# Dummy classes to simulate Azure ML client behavior.
class DummyWorkspace:
    def __init__(self, uri):
        self.mlflow_tracking_uri = uri

class DummyWorkspaces:
    def get(self, workspace_name):
        # Return a dummy workspace with a known mlflow_tracking_uri.
        return DummyWorkspace("dummy_mlflow_uri")

class DummyMLClient:
    def __init__(self, credential, subscription_id, resource_group_name, workspace_name):
        self.credential = credential
        self.subscription_id = subscription_id
        self.resource_group_name = resource_group_name
        self.workspace_name = workspace_name
        self.workspaces = DummyWorkspaces()

# Test for get_mlflow_uri by monkeypatching MLClient and DefaultAzureCredential.
def test_get_mlflow_uri(monkeypatch):
    # Replace MLClient with our dummy implementation.
    monkeypatch.setattr(helpers, "MLClient", DummyMLClient)
    # Replace DefaultAzureCredential with a dummy credential.
    monkeypatch.setattr(helpers, "DefaultAzureCredential", lambda: object())
    
    uri = helpers.get_mlflow_uri()
    assert uri == "dummy_mlflow_uri", "MLflow URI does not match expected dummy value."

# Test for get_tensorboard_writer.
def test_get_tensorboard_writer():
    writer = helpers.get_tensorboard_writer()
    # Check that the writer has a method typically present on TensorBoard writers.
    assert hasattr(writer, "as_default"), "Returned object is not a valid TensorBoard writer."
    # Optionally, you can check if the writer's log directory is as expected.
    expected_log_dir = os.path.abspath("./outputs/segf/tensorboard_logs/")
    # Some writer implementations may expose a "logdir" attribute.
    if hasattr(writer, "logdir"):
        assert expected_log_dir in writer.logdir, "TensorBoard writer log directory is incorrect."
