import sys
import os
import tensorflow as tf
import pytest
import keras.optimizers

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from params.optimizers import unet_optimizer, segf_optimizer

def get_lr_value(optimizer):
    lr = optimizer.learning_rate
    # Depending on the TensorFlow version, lr might be a tensor or a Python float.
    if hasattr(lr, "numpy"):
        return lr.numpy()
    else:
        return tf.keras.backend.get_value(lr)

def test_unet_optimizer():
    optimizer = unet_optimizer()
    assert isinstance(optimizer, keras.optimizers.Adam), "unet_optimizer did not return an Adam optimizer."
    lr_value = get_lr_value(optimizer)
    assert abs(lr_value - 0.0001) < 1e-8, "unet_optimizer learning rate is not 0.0001."

def test_segf_optimizer_default():
    optimizer = segf_optimizer()
    assert isinstance(optimizer, keras.optimizers.Adam), "segf_optimizer did not return an Adam optimizer with default learning rate."
    lr_value = get_lr_value(optimizer)
    assert abs(lr_value - 0.0001) < 1e-8, "segf_optimizer default learning rate is not 0.0001."

def test_segf_optimizer_custom_lr():
    custom_lr = 0.001
    optimizer = segf_optimizer(learning_rate=custom_lr)
    assert isinstance(optimizer, keras.optimizers.Adam), "segf_optimizer did not return an Adam optimizer with custom learning rate."
    lr_value = get_lr_value(optimizer)
    assert abs(lr_value - custom_lr) < 1e-8, "segf_optimizer custom learning rate is not set correctly."
