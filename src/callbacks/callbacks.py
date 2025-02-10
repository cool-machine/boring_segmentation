# src/callbacks/custom_history_factory.py
import os
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import shutil
import mlflow
from datetime import datetime

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau


# 1 History Callback
# Class definition
class CustomHistory(Callback):
    """
    Custom callback to track additional metrics during training.

    This callback maintains a history dictionary to record the values of
    specified metrics at the end of each epoch.

    Attributes:
        history (dict): Dictionary containing recorded metrics.

    Methods:
        on_train_begin: Initializes the history dictionary at the start of training.
        on_epoch_end: Updates the history dictionary at the end of each epoch.
    """

    def on_train_begin(self, logs=None):
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'dice_coefficient': [],
            'val_dice_coefficient': [],
            'iou': [],
            'val_iou': [],
            'during_training_accuracy':[],
            'during_training_iou':[],
            'during_training_dice':[],
        }

    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
        print(f"Epoch {epoch + 1}: {logs}")


# 2 Early Stopping
def create_early_stopping(patience=15, monitor='val_loss', restore_best_weights=True):
    """
    Factory function to create an EarlyStopping callback.

    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        monitor (str): Metric to monitor for early stopping (e.g., 'val_loss', 'val_accuracy').
        restore_best_weights (bool): Whether to restore the model weights from the epoch 
                                     with the best monitored value.

    Returns:
        keras.callbacks.EarlyStopping: Configured EarlyStopping callback.
    """
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=restore_best_weights,
        verbose=1
    )


# 3 Plot Results Callback
# 3.1 Class Definition
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


# 3.2 Function definition for results plotting callback
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


#3.3 Plot results - plot colored segmentation
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


# 4. Learning rate decay callback
def create_reduce_lr(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    cooldown=0,
    verbose=1
):
    """
    Factory function to create a ReduceLROnPlateau callback.

    Args:
        monitor (str): Metric to monitor for reducing learning rate.
        factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        min_lr (float): Lower bound on the learning rate.
        cooldown (int): Number of epochs to wait before resuming normal operation after learning rate is reduced.
        verbose (int): Verbosity mode, 0 = silent, 1 = progress messages.

    Returns:
        keras.callbacks.ReduceLROnPlateau: Configured ReduceLROnPlateau callback.
    """
    return ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        cooldown=cooldown,
        verbose=verbose
    )


# 4. Checkpoint callback - saving K best models
best_models = []
def maybe_save_best_model(model, epoch_val_loss, epoch):
    global best_models
    
    if len(best_models) < 3 or epoch_val_loss < max(m["val_loss"] for m in best_models):
        model_dir = "./outputs/segformer"
        os.makedirs(model_dir, exist_ok=True)

        # Create a *unique* subdir for each checkpoint
        checkpoint_dir = os.path.join(model_dir, f"epoch_{epoch}_val_{epoch_val_loss}")
        # Create the subfolder for this checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Now save_pretrained into this *folder*
        model.save_pretrained(checkpoint_dir)

        # Log the entire directory to MLflow (zips it as an artifact)
        mlflow.log_artifacts(checkpoint_dir, artifact_path=f"checkpoints/epoch_{epoch}")

        # Add the directory path to best_models
        best_models.append({
            "val_loss": epoch_val_loss,
            "epoch": epoch+1,
            "path": checkpoint_dir  # store the folder path
        })

        # Sort & remove the worst
        best_models.sort(key=lambda x: x["val_loss"])
        if len(best_models) > 3:
            to_remove = best_models.pop()
            if os.path.exists(to_remove["path"]):
                # Remove the entire checkpoint folder
                shutil.rmtree(to_remove["path"])


# 4.2 Class definition for saving top K models
class TopKModelCheckpoint(Callback):
    """
    A custom Keras callback to save the top K models during training based on a monitored metric.

    Args:
        filepath (str): Template for the path where model files will be saved. Example: 'model-epoch{epoch:02d}-{val_loss:.4f}.h5'.
        monitor (str): Metric to monitor for selecting the top models (e.g., 'val_loss', 'val_accuracy').
        mode (str): One of {'auto', 'min', 'max'}.
                   - 'min': Save models with the lowest metric values.
                   - 'max': Save models with the highest metric values.
                   - 'auto': Automatically detect based on the metric name.
        top_k (int): Number of top models to retain.
    """

    def __init__(self, filepath, monitor='val_loss', mode='auto', top_k=3):
        super(TopKModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.top_k = top_k
        self.best_models = []  # List to track the best models (metric_value, epoch, filepath)

        if mode not in ['auto', 'min', 'max']:
            print(f"Warning: Mode '{mode}' is unknown. Defaulting to 'auto'.")
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            # Automatically determine the operation based on the metric name
            if 'acc' in self.monitor or 'accuracy' in self.monitor or 'auc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            print(f"Warning: Metric '{self.monitor}' not found in logs. Skipping model saving.")
            return

        # Build model info tuple
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        model_info = (current, epoch + 1, filepath)

        # Determine if we should save the model
        save_model = False

        if len(self.best_models) < self.top_k:
            save_model = True
            self.best_models.append(model_info)
        else:
            # Check if the current model is better than the worst in best_models
            if self.monitor_op(current, self.best_models[-1][0]):
                # Remove the worst model
                worst_model = self.best_models.pop()
                worst_filepath = worst_model[2]
                if os.path.exists(worst_filepath):
                    os.remove(worst_filepath)
                save_model = True
                self.best_models.append(model_info)

        if save_model:
            # Sort the best_models list
            self.best_models.sort(key=lambda x: x[0], reverse=self.monitor_op == np.greater)

            # Save the model
            self.model.save(filepath, overwrite=True)
            print(f"Saved model at {filepath}")


# 5. Saving top k checkpoints
def create_top_k_checkpoint(
    checkpoint_dir='outputs/checkpoints',
    top_k=3,
    monitor='val_loss',
    mode='min'
):
    """
    Factory function to create a custom TopKModelCheckpoint callback.

    Args:
        checkpoint_dir (str): Directory where checkpoint files will be saved.
        top_k (int): Number of top-performing models to save.
        monitor (str): Metric to monitor for saving the best models (e.g., 'val_loss', 'val_accuracy').
        mode (str): One of {'min', 'max'}. In 'min' mode, a lower monitored value is better; 
                    in 'max' mode, a higher monitored value is better.

    Returns:
        src.callbacks.TopKModelCheckpoint: Configured TopKModelCheckpoint callback.
    """
    
    # current_time = datetime.now().strftime("%M-%H-%d-%m-%Y")
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, "model-epoch{epoch:02d}-{val_loss:.4f}.keras")
    return TopKModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        mode=mode,
        top_k=top_k
    )
