import tensorflow as tf
import keras


# Custom Dice Coefficient
def dice_coefficient(y_true, y_pred):
    # One-hot encode the true labels if necessary (assuming y_true is not already one-hot encoded)
    y_true_f = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])

    # Flatten the tensors
    y_true_f = tf.reshape(y_true_f, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # Calculate the intersection
    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    # Calculate the Dice coefficient
    return (2. * intersection + 1) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1)



# Custom IoU (Intersection over Union)
def iou(y_true, y_pred):

    # One-hot encode the true labels if necessary (assuming y_true is not already one-hot encoded)
    y_true_f = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])

    # Flatten the tensors
    y_true_f = tf.reshape(y_true_f, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # Calculate the intersection
    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    # Calculate the union
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    # Calculate the IoU
    return (intersection + 1) / (union + 1)
