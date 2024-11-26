import keras 
from keras.optimizers import Adam


def keras_optimizer():
    return keras.optimizers.Adam(learning_rate=0.0001)