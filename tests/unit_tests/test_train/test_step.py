import sys
import os
import pytest
import tensorflow as tf
from tensorflow import keras
from collections import namedtuple

# Append the src directory to the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from training.step import step

# Dummy metric to capture update_state calls.
class DummyMetric:
    def __init__(self):
        self.last_update = None
    def update_state(self, y_true, y_pred):
        self.last_update = (y_true, y_pred)

# Use a namedtuple for the dummy output.
DummyOutput = namedtuple("DummyOutput", ["logits"])

# Updated DummyModel returns an instance of DummyOutput.
class DummyModel(tf.Module):
    def __init__(self):
        super().__init__()
        # Create a trainable variable.
        self.var = tf.Variable(1.0, trainable=True)

    @tf.function
    def __call__(self, images, training):
        # images shape: [batch, height, width, channels]
        batch = tf.shape(images)[0]
        h = tf.shape(images)[1]
        w = tf.shape(images)[2]
        # Create dummy logits that depend on self.var.
        # Expected shape: [batch, num_classes, height, width]
        logits = tf.ones((batch, 8, h, w)) * self.var
        # Return a namedtuple with the logits tensor.
        return DummyOutput(logits)

# Dummy loss function that returns the mean of logits.
def dummy_loss_fn(y_true, y_pred):
    # This loss depends on the model's output so that gradients are nonzero.
    return tf.reduce_mean(y_pred)

# Fixtures for dummy inputs, model, optimizer, and metrics.
@pytest.fixture
def dummy_inputs():
    # Images: [batch, height, width, channels]
    images = tf.ones((2, 64, 64, 3))
    # Masks: [batch, height, width, 1] (will be squeezed inside step)
    masks = tf.ones((2, 64, 64, 1))
    return images, masks

@pytest.fixture
def dummy_model():
    return DummyModel()

@pytest.fixture
def dummy_optimizer():
    return tf.keras.optimizers.SGD(learning_rate=0.1)

@pytest.fixture
def metrics():
    total_loss_metric = tf.keras.metrics.Mean()
    accuracy_metric = DummyMetric()
    return total_loss_metric, accuracy_metric

def test_step_training(dummy_inputs, dummy_model, dummy_optimizer, metrics):
    images, masks = dummy_inputs
    total_loss_metric, accuracy_metric = metrics

    # Record the initial value of the model variable.
    initial_val = dummy_model.var.numpy()

    # Call step with training=True.
    loss = step(
        images, masks, dummy_model, total_loss_metric, accuracy_metric,
        dummy_loss_fn, dummy_optimizer, training=True
    )
    
    # Check that loss is a scalar tensor.
    assert isinstance(loss, tf.Tensor)
    
    # Check that total_loss metric has been updated with the loss value.
    metric_val = total_loss_metric.result().numpy()
    assert abs(metric_val - loss.numpy()) < 1e-6, "Total loss metric not updated correctly."

    # Check that the DummyMetric captured an update.
    assert accuracy_metric.last_update is not None, "Accuracy metric was not updated."

    # Verify that the optimizer updated the model variable.
    updated_val = dummy_model.var.numpy()
    assert updated_val < initial_val, "Model variable was not updated during training."

def test_step_evaluation(dummy_inputs, dummy_model, dummy_optimizer, metrics):
    images, masks = dummy_inputs
    total_loss_metric, accuracy_metric = metrics

    # Reset metric states using the correct method.
    total_loss_metric.reset_state()
    accuracy_metric.last_update = None

    # Call step with training=False.
    loss = step(
        images, masks, dummy_model, total_loss_metric, accuracy_metric,
        dummy_loss_fn, dummy_optimizer, training=False
    )
    
    # Check that loss is a scalar tensor.
    assert isinstance(loss, tf.Tensor)
    
    # Verify that total_loss metric equals the returned loss.
    metric_val = total_loss_metric.result().numpy()
    assert abs(metric_val - loss.numpy()) < 1e-6, "Total loss metric not updated correctly in evaluation mode."

    # Verify that the DummyMetric captured an update.
    assert accuracy_metric.last_update is not None, "Accuracy metric was not updated in evaluation mode."
