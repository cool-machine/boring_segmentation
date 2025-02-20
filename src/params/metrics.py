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
    y_true_f = tf.one_hot(tf.squeeze(tf.cast(y_true, tf.uint8), axis=-1), depth=tf.cast(8,tf.int32))

    # Flatten the true and predicted labels for element-wise operations
    y_true_f = tf.reshape(y_true_f, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # Ensure both are float32
    y_true_f = tf.cast(y_true_f, tf.uint8)
    y_pred_f = tf.cast(y_pred_f, tf.uint8)

    # tf.print("masks shape in function:", tf.shape(y_true_f))
    # tf.print("predictions in function shape:", tf.shape(y_pred_f))

    # Calculate intersection of true and predicted labels
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    intersection = tf.cast(intersection, tf.float32)

    # Compute the Dice Coefficient
    # nominator
    nominator = (2. * intersection + 1.)
    # denominator
    denominator = tf.cast(tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f), tf.float32) + 1e-7 
    denominator = tf.cast(denominator, tf.float32) + 1e-7

    return  nominator / denominator


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
    # Convert true labels [batch_size, h, w] to one-hot encoding [batch_size, h, w, num_classes]
    
    y_true_f = tf.one_hot(tf.cast(y_true, tf.uint8), depth=tf.cast(8, tf.int32))

    # Flatten the true and predicted labels for element-wise operations

    y_true_f = tf.reshape(y_true_f, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # Ensure both are float32
    y_true_f = tf.cast(y_true_f, tf.uint8)
    y_pred_f = tf.cast(y_pred_f, tf.uint8)

    # Calculate intersection of true and predicted labels
    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    # Calculate union of true and predicted labels
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    # Compute the IoU
    # compute nominator
    nominator = tf.cast(intersection, tf.float32) + 1e-7
    # compute denominator
    denominator = tf.cast(union, tf.float32) + 1e-7 

    return  nominator / denominator

# # Custom IoU (Intersection over Union)
# def iou(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
#     """
#     Calculate Intersection over Union (IoU).
    
#     Args:
#         y_true: Ground truth masks
#         y_pred: Predicted masks (logits)
#         smooth: Smoothing factor to avoid division by zero
    
#     Returns:
#         IoU score (between 0 and 1)
#     """
#     y_pred = tf.nn.softmax(y_pred, axis=-1)  # Convert logits to probabilities
#     y_true = tf.cast(y_true, tf.float32)
    
#     intersection = tf.reduce_sum(y_true * y_pred)
#     union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    
#     iou = (intersection + smooth) / (union + smooth)
#     return iou
#     # return tf.clip_by_value(iou, 0.0, 1.0)

# def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
#     """
#     Calculate Dice coefficient.
    
#     Args:
#         y_true: Ground truth masks
#         y_pred: Predicted masks (logits)
#         smooth: Smoothing factor to avoid division by zero
    
#     Returns:
#         Dice coefficient (between 0 and 1)
#     """
#     y_pred = tf.nn.softmax(y_pred, axis=-1)  # Convert logits to probabilities
#     y_true = tf.cast(y_true, tf.float32)
    
#     intersection = tf.reduce_sum(y_true * y_pred)
#     union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    
#     dice = (2. * intersection + smooth) / (union + smooth)
#     return dice 
#     # tf.clip_by_value(dice, 0.0, 1.0)