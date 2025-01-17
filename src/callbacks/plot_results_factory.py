# src/callbacks/plot_results_factory.py

import os
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import tensorflow as tf
from keras.callbacks import Callback

class PlotResultsCallback(Callback):
    """
    Custom callback for plotting and saving validation results during training.

    This callback generates plots of the original images, true masks, and predicted masks
    at regular intervals during training.

    Args:
        validation_data (tf.data.Dataset): Validation dataset for generating predictions.
        plot_interval (int): Number of epochs between plot generations.

    Methods:
        on_epoch_end: Called at the end of each epoch to plot and save results.
    """

    def __init__(self, validation_data, plot_interval=10):
        super(PlotResultsCallback, self).__init__()
        self.validation_data = validation_data
        self.plot_interval = plot_interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.plot_interval == 0:
            # Get a batch of validation data
            val_images, val_masks = next(iter(self.validation_data))

            # Predict the masks
            predictions = self.model(val_images, training=False)
            predicted_masks = K.argmax(predictions, axis=-1)

            # Create and save plots
            os.makedirs('outputs/plots', exist_ok=True)
            plot_filename = f'epoch_{epoch}_plot.png'
            plot_path = os.path.join('outputs/plots', plot_filename)

            plt.figure(figsize=(12, 4))

            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(val_images[0])
            plt.title("Image")
            plt.axis("off")

            # True mask
            plt.subplot(1, 3, 2)
            plt.imshow(val_images[0])
            plt.imshow(tf.squeeze(val_masks[0]), cmap='jet', alpha=0.5)
            plt.title("True Mask")
            plt.axis("off")

            # Predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(val_images[0])
            plt.imshow(tf.squeeze(predicted_masks[0]), cmap='jet', alpha=0.5)
            plt.title("Predicted Mask")
            plt.axis("off")

            plt.savefig(plot_path)
            plt.close()
            

def plot_segmentation_results(image, true_mask, pred_mask):
    """
    Utility to plot a single image, its true mask, and predicted mask.
    """
    
    # image, true_mask, pred_mask = tf.transpose(image, [1, 2, 0]), tf.transpose(true_mask, [1,2,0]), tf.transpose(pred_mask, [1,2,0])
    
    image = tf.transpose(image, [1, 2, 0])
    pred_mask = tf.argmax(pred_mask, axis = -1)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Convert from Tensor to NumPy array (and possibly reshape for plotting)
    image_np = image.numpy()
    true_mask_np = true_mask.numpy()
    pred_mask_np = pred_mask.numpy()
    print(f"image shape : {image_np.shape}")
    print(f"true mask shape: {true_mask_np.shape}")
    print(f"predicted mask shape: {pred_mask_np.shape}")
    
    # If the image has 3 channels and shape (H,W,3), show it directly:
    axes[0].imshow(image_np.astype("uint8"))  # or handle float range
    axes[0].set_title("Original Image")
    
    axes[1].imshow(true_mask_np, cmap="jet")
    axes[1].set_title("True Mask")
    
    axes[2].imshow(pred_mask_np, cmap="jet")
    axes[2].set_title("Predicted Mask")
    
    for ax in axes:
        ax.axis("off")
    
    plt.tight_layout()
    return fig