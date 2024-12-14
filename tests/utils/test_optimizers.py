from src.utils.optimizers import unet_optimizer
import keras

def test_unet_optimizer():
    opt = unet_optimizer()
    assert isinstance(opt, keras.optimizers.Adam)
    assert opt.learning_rate == 0.0001
