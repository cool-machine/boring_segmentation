import sys
import os
import tensorflow as tf
import numpy as np
import pytest

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from params.metrics import cust_accuracy, dice_coefficient, iou

def test_cust_accuracy():
    # For a perfect match, we expect 100% accuracy.
    # Here we generate one-hot encoded predictions for a perfect match.
    y_true = tf.constant([1, 0, 2])
    y_pred = tf.one_hot(y_true, depth=3, on_value=1.0, off_value=0.0)
    accuracy = cust_accuracy(y_true, y_pred)
    np.testing.assert_allclose(accuracy.numpy(), np.array([1, 1, 1]), atol=1e-5)

def test_dice_coefficient():
    # For dice_coefficient, test with a "perfect match" case.
    # Define y_true as integer labels (shape: [batch, 1]) and create matching one-hot predictions.
    y_true = tf.constant([[1], [2], [3], [4]])
    # Convert y_true to one-hot to simulate perfect predictions.
    y_pred = tf.one_hot(tf.squeeze(y_true, axis=-1), depth=8)
    dice_val = dice_coefficient(y_true, y_pred).numpy()

    # For a mismatched case, use a constant prediction that doesn't match y_true.
    y_pred_mismatch = tf.one_hot(tf.zeros_like(tf.squeeze(y_true, axis=-1)), depth=8)
    dice_val_mismatch = dice_coefficient(y_true, y_pred_mismatch).numpy()

    # The dice coefficient for a perfect match should be higher than that of a mismatch.
    assert dice_val > dice_val_mismatch

def test_iou():
    # For iou, test with a "perfect match" case.
    y_true = tf.constant([[1], [2], [3], [4]])
    y_pred = tf.one_hot(tf.squeeze(y_true, axis=-1), depth=8)
    iou_val = iou(y_true, y_pred).numpy()

    # For a mismatched case, use a constant prediction.
    y_pred_mismatch = tf.one_hot(tf.zeros_like(tf.squeeze(y_true, axis=-1)), depth=8)
    iou_val_mismatch = iou(y_true, y_pred_mismatch).numpy()

    # IoU should be higher for perfect match than for mismatch.
    assert iou_val > iou_val_mismatch
