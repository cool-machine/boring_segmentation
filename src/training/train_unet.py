# Import necessary libraries
import os
import sys
import keras
import mlflow

# Dynamically add the `img_segmentation` root directory to `sys.path`
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
print(f"This is the root directory: {root_dir}")

# Import custom callbacks and utilities
from src.callbacks import (
    create_early_stopping,
    create_reduce_lr,
    create_top_k_checkpoint,
    PlotResultsCallback,
    CustomHistory,
)

from src.data_processing.process_data import load_dataset_unet
from src.utils.metrics import cust_accuracy, dice_coefficient, iou
from src.architectures.unet_model import unet_with_vgg16_encoder
from src.utils.optimizers import unet_optimizer
from src.utils.helpers import get_mlflow_uri


# Define the main training function
def main():
    """
    Main function to initialize, compile, and train the U-Net model with VGG16 encoder.
    Integrates MLflow for experiment tracking.
    """
    # Clear Keras backend to ensure a fresh session
    keras.backend.clear_session()

    # Set TensorFlow deterministic behavior
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # Configure MLflow tracking URI
    mlflow_tracking_uri = get_mlflow_uri()
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Set the experiment name in MLflow
    experiment_name = "unet_experiment_v101"
    mlflow.set_experiment(experiment_name)

    # Load training and validation datasets
    dataset_train, dataset_val = load_dataset_unet(batch_size=1)


    length_tr = sum([1 for _ in dataset_train])
    length_vl = sum([1 for _ in dataset_val])
    print()
    print(f"training dataset contains {length_tr} entries")
    print(f"validation dataset contains {length_vl} entries")
    print()
#     # Initialize callbacks
#     plot_results = PlotResultsCallback(validation_data=dataset_val, plot_interval=5)
#     callbacks = [
#         create_early_stopping(patience=15),
#         create_reduce_lr(patience=10),
#         create_top_k_checkpoint(checkpoint_dir='outputs/checkpoints', top_k=3),
#         CustomHistory(),
#         plot_results,
#     ]

#     # Define model parameters
#     metrics = [cust_accuracy, dice_coefficient, iou]
#     input_shape = (1024, 2048, 3)
#     num_classes = 8

#     # Initialize and compile the model
#     model = unet_with_vgg16_encoder(input_shape, num_classes)
#     optimizer = unet_optimizer()
#     model.compile(
#         optimizer=optimizer,
#         loss='sparse_categorical_crossentropy',
#         metrics=metrics
#     )

#     mlflow.keras.autolog()  # Enable automatic logging of model metrics and parameters

#     # Train the model and log with MLflow
#     with mlflow.start_run():
 
#         history = model.fit(
#             dataset_train.take(1),  # Train on one batch for testing
#             validation_data=dataset_val.take(1),  # Validate on one batch for testing
#             epochs=1,
#             callbacks=callbacks,
#         )


# Entry point for the script
if __name__ == "__main__":
    # Log script execution start
    print("\n\n")
    print("*" * 60)
    print("Starting U-Net Training with MLflow")
    print("*" * 60)
    print("\n\n")

    # Execute the main function
    main()

    # Log script execution end
    print("\n\n")
    print("*" * 60)
    print("Training Complete")
    print("*" * 60)
    print("\n\n")