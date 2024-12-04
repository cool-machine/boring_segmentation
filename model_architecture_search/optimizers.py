
import keras

def unet_optimizer():
    return keras.optimizers.Adam(learning_rate=0.0001)