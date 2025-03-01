import os
import sys
import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add the project root to sys.path so that imports work.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

from src.data.processor import (
    read_image,
    read_label,
    normalize,
    retrieve_mask_mappings,
    map_labels_tf,
    augment_image_and_label,
    resize_images,
    _prepare_dataset,
    load_dataset_unet,
    load_dataset_segf
)

# ---------------------------
# Test read_image and read_label
# ---------------------------

@pytest.fixture
def dummy_image_file(tmp_path):
    # Create a dummy image of shape (1024,2048,3) with random uint8 values.
    img_array = np.random.randint(0, 256, (1024, 2048, 3), dtype=np.uint8)
    # Encode as PNG.
    img_tensor = tf.convert_to_tensor(img_array)
    encoded = tf.io.encode_png(img_tensor).numpy()
    file_path = tmp_path / "dummy_image.png"
    file_path.write_bytes(encoded)
    return str(file_path)

@pytest.fixture
def dummy_label_file(tmp_path):
    # Create a dummy label of shape (1024, 2048, 1) with random uint8 values.
    label_array = np.random.randint(0, 256, (1024, 2048, 1), dtype=np.uint8)
    label_tensor = tf.convert_to_tensor(label_array)
    # Do not squeeze, so the shape remains (1024, 2048, 1)
    encoded = tf.io.encode_png(label_tensor).numpy()
    file_path = tmp_path / "dummy_label.png"
    file_path.write_bytes(encoded)
    return str(file_path)

def test_read_image(dummy_image_file):
    image = read_image(dummy_image_file)
    # Check shape and dtype.
    assert image.shape == (1024, 2048, 3)
    assert image.dtype == tf.float32
    # Pixel values should be in [0,1].
    assert tf.reduce_max(image).numpy() <= 1.0
    assert tf.reduce_min(image).numpy() >= 0.0

def test_read_label(dummy_label_file):
    label = read_label(dummy_label_file)
    # Check shape and dtype.
    assert label.shape == (1024, 2048, 1)
    assert label.dtype == tf.uint8

# ---------------------------
# Test normalize
# ---------------------------
def test_normalize():
    # Create a dummy image and mask.
    dummy_img = tf.ones((512, 512, 3), dtype=tf.float32) * 0.5
    dummy_mask = tf.zeros((512, 512, 1), dtype=tf.uint8)
    norm_img, norm_mask = normalize(dummy_img, dummy_mask)
    # Ensure normalized image is clipped between 0 and 1.
    assert tf.reduce_max(norm_img).numpy() <= 1.0
    assert tf.reduce_min(norm_img).numpy() >= 0.0
    # Mask should be unchanged.
    np.testing.assert_array_equal(norm_mask.numpy(), dummy_mask.numpy())

# ---------------------------
# Test retrieve_mask_mappings and map_labels_tf
# ---------------------------
def test_retrieve_mask_mappings():
    original_classes, class_mapping, new_labels = retrieve_mask_mappings()
    # Check that expected keys exist in new_labels.
    for key in ['flat', 'human', 'vehicle', 'construction', 'object', 'nature', 'sky', 'void']:
        assert key in new_labels

def test_map_labels_tf():
    original_classes, class_mapping, new_labels = retrieve_mask_mappings()
    # Create a dummy label image of shape (1024,2048,1) filled with the index for 'road'
    # In original_classes, 'road' is at index 0.
    dummy_label = tf.fill([1024, 2048, 1], 0)
    mapped = map_labels_tf(dummy_label, original_classes, class_mapping, new_labels)
    # For 'road', class_mapping maps to 'flat', and new_labels['flat'] is expected (typically 0).
    expected_value = new_labels['flat']
    # Check that the mapped label tensor is filled with the expected value.
    np.testing.assert_array_equal(mapped.numpy(), np.full((1024, 2048, 1), expected_value, dtype=np.uint8))

# ---------------------------
# Test augment_image_and_label
# ---------------------------
def test_augment_image_and_label_no_augmentation():
    # Force no augmentation by setting augment_prob=0.
    dummy_img = tf.ones((512,512,3), dtype=tf.float32)
    dummy_mask = tf.zeros((512,512,1), dtype=tf.uint8)
    # Monkey-patch tf.random.uniform to always return 1 so that condition (random < 0) is False.
    original_uniform = tf.random.uniform
    tf.random.uniform = lambda *args, **kwargs: tf.constant(1.0)
    out_img, out_mask = augment_image_and_label(dummy_img, dummy_mask, augment_prob=0.0)
    tf.random.uniform = original_uniform  # Restore
    # Output should equal input.
    np.testing.assert_array_equal(out_img.numpy(), dummy_img.numpy())
    np.testing.assert_array_equal(out_mask.numpy(), dummy_mask.numpy())

# ---------------------------
# Test resize_images
# ---------------------------
def test_resize_images():
    # Create dummy image and label with arbitrary shape.
    dummy_img = tf.ones((600,800,3), dtype=tf.float32)
    dummy_label = tf.ones((600,800,1), dtype=tf.uint8)
    resized_img, resized_label = resize_images(dummy_img, dummy_label)
    # After resizing and transposition, expected shapes are [3,512,1024] and [1,128,256]
    assert resized_img.shape == (3, 512, 1024)
    assert resized_label.shape == (1, 128, 256)

# ---------------------------
# Test _prepare_dataset
# ---------------------------
def test_prepare_dataset(monkeypatch):
    # Prepare dummy lists of file paths.
    dummy_image_paths = ["dummy_img1.png", "dummy_img2.png"]
    dummy_mask_paths = ["dummy_mask1.png", "dummy_mask2.png"]
    
    # Monkey-patch read_image and read_label to return constant tensors.
    monkeypatch.setattr("src.data.processor.read_image", lambda fp: tf.zeros((1024,2048,3), dtype=tf.float32))
    monkeypatch.setattr("src.data.processor.read_label", lambda fp: tf.zeros((1024,2048,1), dtype=tf.uint8))
    # Also bypass map_labels_tf to just return the mask.
    monkeypatch.setattr("src.data.processor.map_labels_tf", lambda label, a, b, c: label)
    # And bypass normalization to return inputs as-is.
    monkeypatch.setattr("src.data.processor.normalize", lambda im, m: (im, m))
    # And augmentation: force no augmentation.
    monkeypatch.setattr("src.data.processor.augment_image_and_label", lambda im, m, augment_prob=0.0: (im, m))
    
    ds = _prepare_dataset(dummy_image_paths, dummy_mask_paths, batch_size=2, is_train=True, resize_fn=None)
    # Should have one batch since drop_remainder=True.
    batch = next(iter(ds))
    # batch should be a tuple: (images, masks)
    images, masks = batch
    assert images.shape[0] == 2
    assert masks.shape[0] == 2

# ---------------------------
# Test load_dataset_unet and load_dataset_segf (monkey-patching get_datasets)
# ---------------------------
@pytest.fixture
def dummy_datasets(tmp_path):
    # Create dummy dataset dict with file paths.
    # For simplicity, each split will have one file.
    return {
        "train_images": [str(tmp_path / "train_img.png")],
        "train_masks": [str(tmp_path / "train_mask.png")],
        "valid_images": [str(tmp_path / "valid_img.png")],
        "valid_masks": [str(tmp_path / "valid_mask.png")],
        "test_images": [str(tmp_path / "test_img.png")],
        "test_masks": [str(tmp_path / "test_mask.png")]
    }

def test_load_dataset_unet(monkeypatch, dummy_datasets):
    # Monkey-patch get_datasets in processor to return dummy_datasets.
    monkeypatch.setattr("src.data.processor.get_datasets", lambda root_path=None: dummy_datasets)
    # Also bypass read_image, read_label, normalization.
    monkeypatch.setattr("src.data.processor.read_image", lambda fp: tf.zeros((1024,2048,3), dtype=tf.float32))
    monkeypatch.setattr("src.data.processor.read_label", lambda fp: tf.zeros((1024,2048,1), dtype=tf.uint8))
    monkeypatch.setattr("src.data.processor.normalize", lambda im, m: (im, m))
    
    train_ds, valid_ds = load_dataset_unet(train_batch_size=1, valid_batch_size=1)
    # Check that datasets are tf.data.Dataset instances and yield a batch.
    train_batch = next(iter(train_ds))
    valid_batch = next(iter(valid_ds))
    assert isinstance(train_batch, tuple)
    assert isinstance(valid_batch, tuple)

def test_load_dataset_segf(monkeypatch, dummy_datasets):
    monkeypatch.setattr("src.data.processor.get_datasets", lambda root_path=None: dummy_datasets)
    monkeypatch.setattr("src.data.processor.read_image", lambda fp: tf.zeros((1024,2048,3), dtype=tf.float32))
    monkeypatch.setattr("src.data.processor.read_label", lambda fp: tf.zeros((1024,2048,1), dtype=tf.uint8))
    monkeypatch.setattr("src.data.processor.normalize", lambda im, m: (im, m))
    # For segf, a resizing function is applied.
    monkeypatch.setattr("src.data.processor.resize_images", lambda im, m: (tf.zeros((3,512,1024)), tf.zeros((1,128,256))))
    
    train_ds, valid_ds, test_ds = load_dataset_segf(train_batch_size=1, valid_batch_size=1)
    train_batch = next(iter(train_ds))
    valid_batch = next(iter(valid_ds))
    test_batch = next(iter(test_ds))
    assert isinstance(train_batch, tuple)
    assert isinstance(valid_batch, tuple)
    assert isinstance(test_batch, tuple)
