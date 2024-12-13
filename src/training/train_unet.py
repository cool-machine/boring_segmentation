# Import libraries
import os
import sys
import keras
# import mlflow

# # Dynamically add the `img_segmentation` root directory to `sys.path`
# root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# if root_dir not in sys.path:
#     sys.path.append(root_dir)
# print(f"this is a root directory: {root_dir}")


from src.callbacks import (
    create_early_stopping,
    create_reduce_lr,
    create_top_k_checkpoint,
    PlotResultsCallback,
    CustomHistory,
)


# Create callbacks
early_stopping = create_early_stopping(patience=15)
reduce_lr = create_reduce_lr(patience=10)
top_k_checkpoint = create_top_k_checkpoint(checkpoint_dir='outputs/checkpoints', top_k=3)
custom_history = CustomHistory()


from src.data_processing.process_data import load_dataset_unet
from src.utils.metrics import cust_accuracy, dice_coefficient, iou

from src.architectures.unet_model import unet_with_vgg16_encoder
from src.utils.optimizers import unet_optimizer
# from logs.mlflow.mlflow_functions import get_mlflow_uri


# define functions
def main():
    # clear Keras backend before compiling and training a new model
    keras.backend.clear_session()    

    # set 'TF_CUDNN_DETERMINISTIC to 1 to avoid approximations (otherwise facing a large errors)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
#     # mlflow set tracking uri
#     mlflow_tracking_uri = get_mlflow_uri()
#     mlflow.set_tracking_uri(mlflow_tracking_uri)
#     # set_tracking_uri()
    dataset_train, dataset_val = load_dataset_unet(batch_size=1)
    plot_results = PlotResultsCallback(validation_data=dataset_val, plot_interval=5)
    callbacks = [early_stopping, reduce_lr, top_k_checkpoint, custom_history, plot_results]
#     # Create an experiment in Azure ML
#     experiment_name = "unet_experiment_v101"
#     mlflow.set_experiment(experiment_name)
    metrics = [cust_accuracy, dice_coefficient, iou] #['accuracy'] #cust_accuracy]
    input_shape = (1024, 2048, 3)
    num_classes = 8

    model = unet_with_vgg16_encoder(input_shape, num_classes)
    optimizer = unet_optimizer()

    model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=metrics)


#     with mlflow.start_run():
#         mlflow.keras.autolog()
        
    history = model.fit(
                dataset_train.take(1),
                validation_data=dataset_val.take(1),
                epochs=1,
                callbacks=callbacks,
                )



# # def evaluate_unet(model, data):
# #     all_metrics = []
# #     for imgs in data:
# #         predictions = model.predict(data['images'])
# #         metric1 = compute_metrics1(predictions, data['labels'])
# #         metrics2 = compute_metrics2(predictions)
# #         all_metrics.append([metric1, metric2])    



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