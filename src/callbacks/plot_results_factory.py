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
            os.makedirs('plots', exist_ok=True)
            plot_filename = f'epoch_{epoch}_plot.png'
            plot_path = os.path.join('plots', plot_filename)

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
            plt.imshow(predicted_masks[0], cmap='jet', alpha=0.5)
            plt.title("Predicted Mask")
            plt.axis("off")

            plt.savefig(plot_path)
            plt.close()



# # custom made plot callback extending Keras Callback class
# class PlotResultsCallback(keras.callbacks.Callback):
#     def __init__(self, validation_data, plot_interval = 10):
#         super(PlotResultsCallback, self).__init__()
#         self.validation_data = validation_data
#         self.plot_interval = plot_interval

#     def on_epoch_end(self, epoch, logs=None):
#         # Plot every 'plot_interval' epochs
#         if epoch % self.plot_interval == 0:
#             # Get a batch of validation data
#             val_images, val_masks = next(iter(self.validation_data))

#             # Predict the masks
#             predictions = self.model(val_images, training=False)
#             predicted_masks = K.argmax(predictions, axis=-1)

#             # Plot the results
#             # for i in range(self.num_samples):
#             plt.figure(figsize=(12, 4))

#             # Original image
#             plt.subplot(1, 3, 1)
#             plt.imshow(val_images[0])
#             plt.title("Image")
#             plt.axis("off")

#             # True mask
#             plt.subplot(1, 3, 2)
#             plt.imshow(val_images[0])
#             plt.imshow(tf.squeeze(val_masks[0]),cmap='jet', alpha=0.5)
#             plt.title("True Mask")
#             plt.axis("off")

#             plt.subplot(1, 3, 3)
#             plt.imshow(val_images[0])
#             plt.imshow(predicted_masks[0], cmap='jet', alpha=0.5)
#             plt.title("Predicted Mask")
#             plt.axis("off")
#             plt.show()
            
#             # Save the plot to a file
#             plot_filename = f'epoch_{epoch}_plot.png'
#             plot_path = os.path.join('plots', plot_filename)
#             os.makedirs('plots', exist_ok=True)
#             plt.savefig(plot_path)
#             plt.close()
