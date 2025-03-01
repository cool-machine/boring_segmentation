import os
import sys
import tempfile
import pytest
import tensorflow as tf
import mlflow
from collections import namedtuple

# Add the project root to sys.path so that imports work.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import the train_segformer module from the training folder.
from src.training import train_segformer

# ----- Dummy Segformer Model Override -----
DummySegformerOutput = namedtuple("DummySegformerOutput", ["logits"])

def dummy_segformer(initial=True, path=""):
    class DummySegformerModel:
        def __init__(self):
            # Create a dummy trainable variable so that gradients can be computed.
            self.dummy_var = tf.Variable(1.0, trainable=True)
            self.trainable_variables = [self.dummy_var]
        def __call__(self, images, training):
            # images are expected to be of shape [batch, 3, 512, 1024] (channels-first).
            batch_size = tf.shape(images)[0]
            # Return logits of shape [batch, 8, 512, 1024] that depend on the dummy variable.
            dummy_logits = tf.ones((batch_size, 8, 512, 1024), dtype=tf.float32) * self.dummy_var
            return DummySegformerOutput(logits=dummy_logits)
        def save_pretrained(self, path):
            # Dummy save: simply create the target directory.
            os.makedirs(path, exist_ok=True)
    # Return an instance so that __init__ is executed.
    return DummySegformerModel()

# ----- Dummy Dataset Loader Override -----
def dummy_load_dataset_segf(train_batch_size=1, valid_batch_size=1):
    # Return images in channels-first format: [batch, 3, 512, 1024].
    dummy_image = tf.zeros((train_batch_size, 3, 512, 1024), dtype=tf.float32)
    # Return masks in channels-first format: [batch, 1, 512, 1024].
    dummy_mask = tf.zeros((train_batch_size, 1, 512, 1024), dtype=tf.uint8)
    dummy_train_ds = tf.data.Dataset.from_tensor_slices((dummy_image, dummy_mask)).batch(train_batch_size)
    dummy_valid_ds = tf.data.Dataset.from_tensor_slices((dummy_image, dummy_mask)).batch(valid_batch_size)
    dummy_test_ds = tf.data.Dataset.from_tensor_slices((dummy_image, dummy_mask)).batch(valid_batch_size)
    return dummy_train_ds, dummy_valid_ds, dummy_test_ds

# ----- Dummy MLflow Functions -----
class DummyMLFlowRun:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        pass

def dummy_start_run(*args, **kwargs):
    return DummyMLFlowRun()

def dummy_log_metric(*args, **kwargs):
    pass

def dummy_log_figure(*args, **kwargs):
    pass

# ----- Dummy Training Configuration -----
class DummyTrainingConfig:
    learning_rate = 0.0001
    epochs = 1
    train_batch_size = 1
    valid_batch_size = 1
    early_stopping_patience = 1
    reduce_lr_patience = 1
    model_name = "dummy_segformer"

# ----- Pytest Fixture to Override Dependencies -----
@pytest.fixture(autouse=True)
def override_dependencies(monkeypatch):
    # Override the dataset loader.
    monkeypatch.setattr(train_segformer, "load_dataset_segf", dummy_load_dataset_segf)
    # Override MLflow functions.
    monkeypatch.setattr(mlflow, "start_run", dummy_start_run)
    monkeypatch.setattr(mlflow, "log_metric", dummy_log_metric)
    monkeypatch.setattr(mlflow, "log_figure", dummy_log_figure)
    # Override the TrainingConfig so that the training runs quickly.
    monkeypatch.setattr(train_segformer, "TrainingConfig", lambda: DummyTrainingConfig())
    # Override the segformer function with our dummy implementation.
    monkeypatch.setattr(train_segformer, "segformer", dummy_segformer)
    # Override the resize function in the processor module to be a no-op.
    monkeypatch.setattr("src.data.processor.resize_images", lambda im, m: (im, m))

# ----- Integration Test -----
def test_integration_train_segformer():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.environ["OUTPUT_DIR"] = tmp_dir  # if your training script uses this variable
        try:
            train_segformer.main()
        except Exception as e:
            pytest.fail(f"Integration run of train_segformer.py failed: {e}")

