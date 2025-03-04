import sys
import os
import pytest
from transformers import TFSegformerForSemanticSegmentation, SegformerConfig


import tensorflow as tf
import keras
from keras import layers, models
# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from params.architectures.segformer import segformer

# Dummy model to return from our fake from_pretrained
class DummyModel:
    def __init__(self, pretrained, config, ignore_mismatched_sizes):
        self.pretrained = pretrained
        self.config = config
        self.ignore_mismatched_sizes = ignore_mismatched_sizes

# Dummy replacement for from_pretrained
def dummy_from_pretrained(pretrained_model_name_or_path, config, ignore_mismatched_sizes):
    return DummyModel(pretrained_model_name_or_path, config, ignore_mismatched_sizes)

def test_segformer_initial(monkeypatch):
    # Monkeypatch the from_pretrained method
    monkeypatch.setattr(TFSegformerForSemanticSegmentation, "from_pretrained", dummy_from_pretrained)
    
    # Call segformer with initial=True (default)
    model = segformer(initial=True)
    
    # Check that the returned object is our dummy model and parameters are as expected.
    expected_pretrained = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
    assert isinstance(model, DummyModel)
    assert model.pretrained == expected_pretrained
    assert model.config.num_labels == 8

def test_segformer_non_initial(monkeypatch):
    monkeypatch.setattr(TFSegformerForSemanticSegmentation, "from_pretrained", dummy_from_pretrained)
    
    test_path = "dummy_path"
    model = segformer(initial=False, path=test_path)
    
    # Verify that the function uses the provided path when initial is False.
    assert isinstance(model, DummyModel)
    assert model.pretrained == test_path
    assert model.config.num_labels == 8





# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from params.architectures.unet import unet_with_vgg16_encoder

# Dummy layer class to simulate VGG16 layers
class DummyLayer:
    def __init__(self, name, output):
        self.name = name
        self.output = output
        self.trainable = True

# Dummy VGG16 model that returns dummy layers with expected names and outputs.
class DummyVGG16:
    def __init__(self, input_tensor):
        self.input_tensor = input_tensor
        # Create dummy outputs using Keras Input layers with fixed shapes.
        # For testing, we assume a small input size (e.g., 64x64) to keep things simple.
        self.dummy_layers = {
            "block1_conv2": DummyLayer("block1_conv2", tf.keras.Input(shape=(32, 32, 64))),
            "block2_conv2": DummyLayer("block2_conv2", tf.keras.Input(shape=(16, 16, 128))),
            "block3_conv3": DummyLayer("block3_conv3", tf.keras.Input(shape=(8, 8, 256))),
            "block4_conv3": DummyLayer("block4_conv3", tf.keras.Input(shape=(4, 4, 512))),
            "block5_conv3": DummyLayer("block5_conv3", tf.keras.Input(shape=(2, 2, 512))),
        }
        # The layers list is used to iterate and freeze layers.
        self.layers = list(self.dummy_layers.values())
        
    def get_layer(self, name):
        return self.dummy_layers[name]

# Dummy function to replace keras.applications.VGG16
def dummy_VGG16(*args, **kwargs):
    input_tensor = kwargs.get("input_tensor")
    return DummyVGG16(input_tensor)

def test_unet_with_vgg16_encoder(monkeypatch):
    # Monkeypatch the VGG16 function so that our dummy is used instead.
    monkeypatch.setattr(keras.applications, "VGG16", dummy_VGG16)

    # For testing, use a small input shape to avoid heavy computation.
    test_input_shape = (64, 64, 3)
    test_num_classes = 2
    model = unet_with_vgg16_encoder(input_shape=test_input_shape, num_classes=test_num_classes)

    # Check that the returned object is a Keras Model.
    assert isinstance(model, models.Model), "unet_with_vgg16_encoder did not return a Keras Model."

    # Verify that the model input shape matches the provided input shape.
    # model.input_shape is typically (None, 64, 64, 3)
    assert model.input_shape[1:] == test_input_shape, f"Expected input shape {test_input_shape}, got {model.input_shape[1:]}."

    # Check that the output layer has the correct number of filters (num_classes).
    # Based on the dummy encoder and the decoder design, the spatial dimensions might differ.
    # In our dummy setup, the decoder upsamples the bottleneck (shape 2x2) back to 32x32.
    output_shape = model.output_shape  # typically (None, 32, 32, test_num_classes)
    assert output_shape[-1] == test_num_classes, f"Expected {test_num_classes} output channels, got {output_shape[-1]}."
