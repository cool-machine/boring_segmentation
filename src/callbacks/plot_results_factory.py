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


def plot_segmentation_results(image, true_mask, pred_mask, alpha=0.6):
    """
    Utility to plot a single image with its true and predicted masks overlaid.

    Args:
        image (tf.Tensor): Original image tensor with shape (H, W, C).
        true_mask (tf.Tensor): Ground-truth mask tensor with shape (H, W) or (H, W, 1).
        pred_mask (tf.Tensor): Predicted mask tensor with shape (H, W, num_classes).
        alpha (float): Transparency for the mask overlay, between 0 (fully transparent) and 1 (fully opaque).
    """

    image = tf.transpose(image, [1, 2, 0])
    pred_mask = tf.argmax(pred_mask, axis = -1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Convert tensors to NumPy arrays
    # Convert from Tensor to NumPy array (and possibly reshape for plotting)
    image_np = image.numpy()
    true_mask_np = true_mask.numpy()
    pred_mask_np = pred_mask.numpy()
    
    
    # image_np = tf.convert_to_tensor(image).numpy()
    # true_mask_np = tf.convert_to_tensor(true_mask).numpy()
    # pred_mask_np = tf.argmax(pred_mask, axis=-1).numpy()

    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the original image
    axes[0].imshow(image_np.astype("float32"))
    axes[0].set_title("Original Image")
    
    # Overlay true mask on the original image
    axes[1].imshow(image_np.astype("float32"))
    axes[1].imshow(true_mask_np, cmap="jet", alpha=alpha)
    axes[1].set_title("True Mask Overlay")
    
    # Overlay predicted mask on the original image
    axes[2].imshow(image_np.astype("float32"))
    axes[2].imshow(pred_mask_np, cmap="jet", alpha=alpha)
    axes[2].set_title("Predicted Mask Overlay")
    
    # Remove axes for a cleaner look
    for ax in axes:
        ax.axis("off")
    
    plt.tight_layout()
    return fig



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_colored_segmentation(image, mask, pred_mask):
    """
    Plot an image with a colorized segmentation mask overlay.
    
    Args:
        image (tf.Tensor): Original image tensor with shape (H, W, C).
        mask (tf.Tensor): Segmentation mask tensor with shape (H, W) containing category indices.
        categories_colors (dict): A mapping of category indices to RGB colors (0-255).
        alpha (float): Transparency for the mask overlay, between 0 (fully transparent) and 1 (fully opaque).
    
    Returns:
        None: Displays the visualization.
    """

    # Define colors for 8 categories (RGB values)
    categories_colors = {
        0: [255, 0, 0],    # Red
        1: [0, 255, 0],    # Green
        2: [0, 0, 255],    # Blue
        3: [255, 255, 0],  # Yellow
        4: [255, 0, 255],  # Magenta
        5: [0, 255, 255],  # Cyan
        6: [128, 128, 128], # Gray
        7: [255, 165, 0],  # Orange
    }

    # Prepare image for plotting
    image = tf.transpose(image, [1, 2, 0])
    
    # Prepare mask for plotting 
    mask = tf.squeeze(mask)
    # mask_pred = "Prediction"

    # if not label:
    #     mask = tf.argmax(mask, axis = -1)
    
    # if label:
    #     mask_name = "Original"

    # Convert image and mask to NumPy arrays
    image_np = image.numpy().astype("float32")
    mask_np = mask.numpy().astype("uint8")
    
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.squeeze(pred_mask)
    mask_np_pred = pred_mask.numpy().astype("uint8")
    
    
    # Create a blank color image for the mask
    color_true_mask = np.zeros((*mask_np.shape, 3), dtype=np.float32)
    
    
    # Create a blank color image for the mask
    color_pred_mask = np.zeros((*mask_np_pred.shape, 3), dtype=np.float32)
 
    # Map each category to its corresponding color
    for category, color in categories_colors.items():
        color_true_mask[mask_np == category] = np.array(color) 

       
    # Map each category to its corresponding color
    for category, color in categories_colors.items():
        color_pred_mask[mask_np_pred == category] = np.array(color) 


    # Overlay the colorized mask on the original image
    # blended = (1 - alpha) * image_np + alpha * color_mask

    # Plot the original image, colorized mask, and blended result
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(image_np)
    axes[1].imshow(color_true_mask, cmap='jet', alpha=0.5)
    axes[1].set_title(f"Ground Truth Colorized Mask")
    axes[1].axis("off")

    axes[2].imshow(image_np)
    axes[2].imshow(color_pred_mask, cmap='jet', alpha=0.5)
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    plt.tight_layout()
    return fig
