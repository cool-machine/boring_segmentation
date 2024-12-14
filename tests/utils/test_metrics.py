import pytest
import tensorflow as tf
from src.utils.metrics import cust_accuracy, dice_coefficient, iou

@pytest.mark.parametrize("y_true, y_pred, expected", [
    ([[1]], [[[0.1, 0.9]]], 1.0),  # Perfect prediction
    ([[0]], [[[0.9, 0.1]]], 1.0),  # Perfect prediction for class 0
    ([[1]], [[[0.8, 0.2]]], 0.0),  # Incorrect prediction
])
def test_cust_accuracy(y_true, y_pred, expected):
    acc = cust_accuracy(tf.constant(y_true), tf.constant(y_pred))
    assert pytest.approx(tf.reduce_mean(acc).numpy(), 0.001) == expected

def test_dice_coefficient_perfect_overlap():
    y_true = tf.constant([1, 1, 0, 0])
    # Two classes: class 0 and class 1
    y_pred = tf.constant([
        [0.0, 1.0],  # matches true class 1
        [0.0, 1.0],  # matches true class 1
        [1.0, 0.0],  # matches true class 0
        [1.0, 0.0]   # matches true class 0
    ], dtype=tf.float32)
    dice = dice_coefficient(y_true, y_pred).numpy()
    assert dice == pytest.approx(1.0, 0.001)

def test_dice_coefficient_no_overlap():
    y_true = tf.constant([1, 1, 1, 1])  # all class 1
    y_pred = tf.constant([
        [1.0, 0.0],  # predicts class 0 only
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0]
    ], dtype=tf.float32)
    dice = dice_coefficient(y_true, y_pred).numpy()
    assert dice < 0.2


def test_iou_perfect_overlap():
    y_true = tf.constant([1, 1, 0, 0])
    # Two classes: class 0 and class 1
    y_pred = tf.constant([
        [0.0, 1.0],  # matches true class 1
        [0.0, 1.0],  # matches true class 1
        [1.0, 0.0],  # matches true class 0
        [1.0, 0.0]   # matches true class 0
        ], dtype=tf.float32)


    metric = iou(y_true, y_pred).numpy()
    assert metric == pytest.approx(1.0, 0.001)

def test_iou_no_overlap():
    y_true = tf.constant([1, 1, 1, 1])  # all class 1
    y_pred = tf.constant([
        [1.0, 0.0],  # predicts class 0 only
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0]
    ], dtype=tf.float32)
    metric = iou(y_true, y_pred).numpy()
    # With smoothing, IoU won't be zero, but it should be small.
    assert metric < 0.2
