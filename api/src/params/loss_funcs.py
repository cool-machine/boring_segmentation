import keras

def sparse_categorical_crossentropy_loss(from_logits=False):
    """
    Returns the Keras sparse categorical cross-entropy loss function.

    This function computes the sparse categorical cross-entropy loss, 
    which is used for multi-class classification problems with integer labels.

    Args:
        from_logits (bool, optional): 
            If True, the model's predictions are raw logits and the 
            softmax operation will be applied internally. 
            Defaults to False.
        axis (int, optional): 
            The axis along which the classes are defined. 
            Defaults to -1 (last axis).

    Returns:
        keras.losses.Loss: 
            A callable sparse categorical cross-entropy loss function.
    
    Example:
        >>> loss_fn = sparse_categorical_crossentropy_loss(from_logits=False)
        >>> y_true = [0, 1, 2]
        >>> y_pred = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]]
        >>> loss = loss_fn(y_true, y_pred)
        >>> print(loss.numpy())
    """
    return keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)