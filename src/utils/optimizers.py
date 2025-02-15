import keras

def unet_optimizer():
    """
    Creates and returns an Adam optimizer with a predefined learning rate.
    
    This optimizer is designed for use with the U-Net architecture or similar
    deep learning models. The Adam optimizer combines the advantages of two
    popular optimization techniques: Adaptive Gradient Algorithm (AdaGrad)
    and Root Mean Square Propagation (RMSProp), making it suitable for 
    tasks like semantic segmentation.

    Learning Rate:
        - A learning rate of 0.0001 is set, which is generally a good 
          starting point for fine-tuning models with moderate complexity like U-Net.

    Returns:
        keras.optimizers.Adam:
            An instance of the Adam optimizer with a specified learning rate.
    """
    return keras.optimizers.Adam(learning_rate=0.0001)


def segf_optimizer(learning_rate=0.0001):
    """
    Creates and returns an Adam optimizer with a predefined learning rate.
    
    This optimizer is designed for use with the U-Net architecture or similar
    deep learning models. The Adam optimizer combines the advantages of two
    popular optimization techniques: Adaptive Gradient Algorithm (AdaGrad)
    and Root Mean Square Propagation (RMSProp), making it suitable for 
    tasks like semantic segmentation.

    Learning Rate:
        - A learning rate of 0.0001 is set, which is generally a good 
          starting point for fine-tuning models with moderate complexity like U-Net.

    Returns:
        keras.optimizers.Adam:
            An instance of the Adam optimizer with a specified learning rate.
    """
    return keras.optimizers.Adam(learning_rate=learning_rate)