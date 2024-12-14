from src.architectures.unet_model import unet_with_vgg16_encoder

def test_unet_with_vgg16_encoder():
    """
    Tests the unet_with_vgg16_encoder function for correctness.
    """
    # Test model creation
    model = unet_with_vgg16_encoder(input_shape=(128, 128, 3), num_classes=4)
    assert model is not None, "Model creation failed."

    # Check input shape
    assert model.input_shape == (None, 128, 128, 3), "Input shape mismatch."

    # Check output shape
    assert model.output_shape == (None, 128, 128, 4), "Output shape mismatch."

    # Print model summary (optional)
    model.summary()