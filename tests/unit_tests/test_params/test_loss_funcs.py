
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))


import pytest
import numpy as np
import keras.losses
from params.loss_funcs import sparse_categorical_crossentropy_loss

def test_sparse_categorical_crossentropy_loss():
    """Test the sparse categorical cross-entropy loss function wrapper."""

    # Check if the function returns a valid Keras loss object
    loss_fn = sparse_categorical_crossentropy_loss(from_logits=False)
    assert isinstance(loss_fn, keras.losses.Loss), "Returned object is not a Keras loss function"

    # Define sample true labels and predicted probabilities
    y_true = np.array([0, 1, 2])  # Ground truth labels
    y_pred = np.array([[0.9, 0.05, 0.05], 
                       [0.1, 0.8, 0.1], 
                       [0.2, 0.3, 0.5]])  # Predicted probabilities

    # Compute loss
    loss_value = loss_fn(y_true, y_pred).numpy()

    # Validate that the loss value is positive (expected behavior)
    assert loss_value > 0, "Loss value should be positive"

    # Test with `from_logits=True`
    loss_fn_logits = sparse_categorical_crossentropy_loss(from_logits=True)
    y_pred_logits = np.log(y_pred + 1e-7)  # Convert probabilities to logits
    loss_value_logits = loss_fn_logits(y_true, y_pred_logits).numpy()

    assert loss_value_logits > 0, "Loss with logits should be positive"

    # Ensure that loss values are close to expected manual computation
    expected_loss = np.mean([-np.log(0.9), -np.log(0.8), -np.log(0.5)])
    assert np.isclose(loss_value, expected_loss, atol=1e-5), "Computed loss does not match expected value"
