import tensorflow as tf
import keras

# Custom Accuracy
def cust_accuracy(y_true, y_pred):
    """
    Computes accuracy using Keras's built-in sparse categorical accuracy metric.
    
    This custom implementation serves as a wrapper to ensure consistency 
    with other custom metrics defined in this module.
    
    Parameters:
        y_true: Tensor
            True labels. Assumes labels are in integer-encoded form (not one-hot encoded).
        y_pred: Tensor
            Predicted probabilities or logits.

    Returns:
        Tensor
            Sparse categorical accuracy score for the given predictions.
    """
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


# Custom Dice Coefficient
def dice_coefficient(y_true, y_pred):
    """
    Computes the Dice Coefficient, a measure of overlap between true and predicted labels.
    Often used in image segmentation tasks.

    The Dice Coefficient is calculated as:
        Dice = (2 * |A ∩ B|) / (|A| + |B|)

    Parameters:
        y_true: Tensor
            True labels. Assumes labels are integer-encoded and not one-hot encoded.
        y_pred: Tensor
            Predicted probabilities or logits.

    Returns:
        Tensor
            The Dice Coefficient score for the given predictions.
    """
    # Convert true labels to one-hot encoding
    y_true_f = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])

    # Flatten the true and predicted labels for element-wise operations
    y_true_f = tf.reshape(y_true_f, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # Calculate intersection of true and predicted labels
    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    # Compute the Dice Coefficient
    return (2. * intersection + 1) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1)


# Custom IoU (Intersection over Union)
def iou(y_true, y_pred):
    """
    Computes the Intersection over Union (IoU) metric, also known as the Jaccard Index.
    IoU measures the overlap between true and predicted labels and is commonly used in 
    tasks like semantic segmentation.

    The IoU is calculated as:
        IoU = |A ∩ B| / |A ∪ B|

    Parameters:
        y_true: Tensor
            True labels. Assumes labels are integer-encoded and not one-hot encoded.
        y_pred: Tensor
            Predicted probabilities or logits.

    Returns:
        Tensor
            The IoU score for the given predictions.
    """
    # Convert true labels to one-hot encoding
    y_true_f = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])

    # Flatten the true and predicted labels for element-wise operations
    y_true_f = tf.reshape(y_true_f, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # Calculate intersection of true and predicted labels
    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    # Calculate union of true and predicted labels
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    # Compute the IoU
    return (intersection + 1) / (union + 1)



# import tensorflow as tf
# import keras


# # Custom Accuracy
# def cust_accuracy(y_true, y_pred):
#     """
#     Wrapper around Keras's built-in accuracy metric.
#     This ensures consistency with custom metrics.
#     """
#     return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


# # Custom Dice Coefficient
# def dice_coefficient(y_true, y_pred):
#     # One-hot encode the true labels if necessary (assuming y_true is not already one-hot encoded)
#     y_true_f = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])

#     # Flatten the tensors
#     y_true_f = tf.reshape(y_true_f, [-1])
#     y_pred_f = tf.reshape(y_pred, [-1])

#     # Calculate the intersection
#     intersection = tf.reduce_sum(y_true_f * y_pred_f)

#     # Calculate the Dice coefficient
#     return (2. * intersection + 1) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1)



# # Custom IoU (Intersection over Union)
# def iou(y_true, y_pred):

#     # One-hot encode the true labels if necessary (assuming y_true is not already one-hot encoded)
#     y_true_f = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])

#     # Flatten the tensors
#     y_true_f = tf.reshape(y_true_f, [-1])
#     y_pred_f = tf.reshape(y_pred, [-1])

#     # Calculate the intersection
#     intersection = tf.reduce_sum(y_true_f * y_pred_f)

#     # Calculate the union
#     union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

#     # Calculate the IoU
#     return (intersection + 1) / (union + 1)


