
# Import libraries
import os
import sys

from azureml.core import Dataset, Workspace

import tensorflow as tf
import keras
from tensorflow.keras import layers, models, backend as K
from tqdm import tqdm

import mlflow
import mlflow.keras
# import mlflow.azureml

from data_processing.image_processing import load_dataset_unet
from callbacks.callbacks import keras_callbacks, PlotResultsCallback
from custom_metrics.custom_metrics import iou, dice_coefficient
from model_definitions.unet import unet_with_vgg16_encoder
from utils.utils import keras_optimizer


# Dynamically add the `img_segmentation` root directory to `sys.path`
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# define functions
def main():

    keras.backend.clear_session()
    # Connect to your Azure ML workspace
    ws = Workspace.from_config()

    # Set the MLflow tracking URI to point to your Azure ML workspace
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    # Create an experiment in Azure ML
    experiment_name = "unet_experiment_v5"
    mlflow.set_experiment(experiment_name)

    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    dataset_train, dataset_val, mount_pts = load_dataset_unet(batch_size=6)
    
    early_stop, model_checkpoint, reduce_lr, custom_history, top_k_checkpoint = keras_callbacks()
    plot_callback=PlotResultsCallback(validation_data=dataset_val)

    callbacks = [early_stop, model_checkpoint, reduce_lr, plot_callback, custom_history,top_k_checkpoint]
    metrics = [dice_coefficient, iou, 'accuracy']

    input_shape = (1024, 2048, 3)  # Input shape for Cityscapes dataset
    num_classes = 8  # Number of major groups/classes in mask
    model = unet_with_vgg16_encoder(input_shape, num_classes)
    optimizer = keras_optimizer()
    model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=metrics)


    with mlflow.start_run():
        mlflow.keras.autolog()
        
        history = model.fit(
                        dataset_train.take(1),
                        validation_data=dataset_val.take(1),
                        epochs=10,
                        callbacks=callbacks)


# run script
if __name__ == "__main__":
    
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # run main function
    main()

    # add space in logs
    print("*" * 60)
    print("\n\n")




# define functions
def main(args):
    
    # read data
    train_dataset = get_train_data(args.training_data)
    valid_dataset = det_valid_data(args.valid_data)
    
    # TO DO: enable autologging

    
    # Combine custom and builtin metrics for Keras model.fit method
    early_stop, model_checkpoint, reduce_lr, plot_callback = keras_callbacks()
    
    callbacks = [early_stop, model_checkpoint, reduce_lr, plot_callback]
    metrics = [dice_coefficient, iou, 'accuracy']

    input_shape = (1024, 2048, 3)  # Input shape for Cityscapes dataset
    num_classes = 8  # Number of major groups/classes in mask
    model = unet_with_vgg16_encoder(input_shape, num_classes)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=metrics)


    # Train the model
    history = model.fit(train_dataset,
                        validation_data=valid_dataset,
                        epochs=1,
                        callbacks=callbacks)

    # split data

    # train model




# load data


# load model
input_shape = (1024, 2048, 3)  # Input shape for Cityscapes dataset
num_classes = 8  # Number of major groups/classes in mask
model = unet_with_vgg16_encoder(input_shape, num_classes)

# define metrics (including for logging)


## Custom History


## TensorBoard


## MLflow



# define callbacks (including for logging)



# Compile the model with the metrics


optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=metrics)

# Train the model
history = model.fit(train_dataset,
                    validation_data=valid_dataset,
                    epochs=250,
                    callbacks=callbacks)

# save the model
model.save('my_model.keras')

# Assuming `history` is the object returned by `model.fit()`
with open('trainHistoryDict.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)