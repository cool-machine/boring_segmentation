import pytest
from data.data_loading import get_datasets

def test_get_datasets():
    datasets = get_datasets()
    assert len(datasets["train_images"]) > 0, "No training images found"
    assert len(datasets["train_masks"]) > 0, "No training masks found"
    assert len(datasets["train_images"]) == len(datasets["train_masks"]), "Mismatch in images and masks"
