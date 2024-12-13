"""
load_data.py

This module provides functions for finding the project root, loading image and mask file paths, 
and verifying that image and mask pairs match correctly.
"""

import os
from pathlib import Path


def load_paths(directory):
    """
    Recursively load all file paths from the provided directory.

    Args:
        directory (str or Path): The directory to walk through.

    Returns:
        list of str: A list of absolute file paths found in the directory.
    """
    full_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            full_path = os.path.abspath(os.path.join(root, file))
            full_paths.append(full_path)
    return full_paths


def find_project_root(current_path, marker_file="README.md"):
    """
    Find the project's root directory by searching upwards until a marker file is found.

    Args:
        current_path (str or Path): The current path from which to start searching upwards.
        marker_file (str): The file used as a marker for the project root.

    Raises:
        FileNotFoundError: If the marker file is not found before reaching the filesystem root.

    Returns:
        Path: The Path object of the project root directory.
    """
    current_path = Path(current_path).resolve()
    while current_path != current_path.root:
        if (current_path / marker_file).exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError(f"Root directory containing {marker_file} not found.")


def check_paths(images, masks):
    """
    Ensure that each image in `images` matches the corresponding mask in `masks` by comparing file names.

    Args:
        images (list of str): A list of image file paths.
        masks (list of str): A list of mask file paths.

    Raises:
        AssertionError: If the number of images doesn't match the number of masks or 
                        if any corresponding image and mask files do not match.
    """
    for img_path, msk_path in zip(images, masks):
        img_name = img_path.split("/")[-1].split("_")[:3]
        msk_name = msk_path.split("/")[-1].split("_")[:3]
        assert img_name == msk_name, (
            f"Paths of image {img_name} and mask {msk_name} do not match"
        )

    assert len(images) == len(masks), (
        f"Number of images {len(images)} does not match the number of masks {len(masks)}"
    )

    print("Paths are correct - check passed")


def get_datasets():
    """
    Retrieve paths for train, validation, and test sets of images and masks.
    Ensure that image-mask pairs match and return a dictionary of datasets.

    Returns:
        dict: A dictionary containing lists of file paths for 'train_images', 
              'train_masks', 'valid_images', 'valid_masks', 'test_images', and 'test_masks'.
    """
    cwd = Path.cwd()
    project_root = find_project_root(cwd)
    print(f"Project root is {project_root}")

    datasets_paths = {}
    dataset_names = [
        "train_images",
        "train_masks",
        "valid_images",
        "valid_masks",
        "test_images",
        "test_masks",
    ]

    # Load training images and masks
    root_train_images = project_root / 'dataset/train/images'
    train_images = load_paths(root_train_images)
    train_images = [img for img in train_images if "_leftImg8bit.png" in img]
    datasets_paths["train_images"] = train_images

    root_train_masks = project_root / 'dataset/train/masks'
    train_masks = load_paths(root_train_masks)
    train_masks = [msk for msk in train_masks if "_labelIds.png" in msk]
    datasets_paths["train_masks"] = train_masks

    check_paths(datasets_paths["train_images"], datasets_paths["train_masks"])

    # Load validation images and masks
    root_valid_images = project_root / 'dataset/valid/images'
    valid_images = load_paths(root_valid_images)
    valid_images = [img for img in valid_images if "_leftImg8bit.png" in img]
    datasets_paths["valid_images"] = valid_images

    root_valid_masks = project_root / 'dataset/valid/masks'
    valid_masks = load_paths(root_valid_masks)
    valid_masks = [msk for msk in valid_masks if "_labelIds.png" in msk]
    datasets_paths["valid_masks"] = valid_masks

    check_paths(datasets_paths["valid_images"], datasets_paths["valid_masks"])

    # Load test images and masks
    root_test_images = project_root / 'dataset/test/images'
    test_images = load_paths(root_test_images)
    test_images = [img for img in test_images if "_leftImg8bit.png" in img]
    datasets_paths["test_images"] = test_images

    root_test_masks = project_root / 'dataset/test/masks'
    test_masks = load_paths(root_test_masks)
    test_masks = [msk for msk in test_masks if "_labelIds.png" in msk]
    datasets_paths["test_masks"] = test_masks

    check_paths(datasets_paths["test_images"], datasets_paths["test_masks"])

    return datasets_paths