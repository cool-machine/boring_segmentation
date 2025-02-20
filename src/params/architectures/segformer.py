# import tensorflow.keras as tf_keras
# from tensorflow.keras import layers, models, backend as K
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
# from tensorflow.keras.optimizers.schedules import ExponentialDecay
# from tensorflow.keras.metrics import AUC
# from tensorflow.keras.optimizers import Adam #SparseCategoricalAccuracy
# from tensorflow.keras.metrics import IoU, Mean, SparseCategoricalAccuracy
# from tensorflow.keras.losses import SparseCategoricalCrossentropy

from transformers import TFSegformerForSemanticSegmentation, SegformerForSemanticSegmentation, SegformerImageProcessor
from transformers import SegformerConfig

# Create a new configuration
def segformer(initial=True, path = ""):
    config = SegformerConfig(
        num_labels=8,  # Set the number of labels/classes
        id2label={0: "flat", 1: "human", 2: "vehicle", 3: "construction", 4: "object", 5: "nature", 6: "sky", 7: "void"},
        label2id={"flat": 0, "human": 1, "vehicle": 2, "construction": 3, "object": 4, "nature": 5, "sky": 6, "void": 7},
        image_size=(512, 1024),  # Specify the input image size
    )
    if initial:
        model = TFSegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-1024", 
                                                                    config=config,
                                                                    ignore_mismatched_sizes=True)
    else: 
        model = TFSegformerForSemanticSegmentation.from_pretrained(path,
                                                                    config=config,
                                                                    ignore_mismatched_sizes=True)
    
    return model

