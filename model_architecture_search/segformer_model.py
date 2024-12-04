
import tensorflow.keras as tf_keras
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam #SparseCategoricalAccuracy
from tensorflow.keras.metrics import IoU, Mean, SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy


