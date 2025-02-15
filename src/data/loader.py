"""
load_data.py

This module provides functions for finding the project root, loading image and mask file paths, 
and verifying that image and mask pairs match correctly.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatasetPaths:
    """Container for dataset paths"""
    train_images: List[str]
    train_masks: List[str]
    valid_images: List[str]
    valid_masks: List[str]
    test_images: List[str]
    test_masks: List[str]

class PathLoader:
    """Handles loading and validation of dataset paths"""
    
    def __init__(self, root_path: Optional[Path] = None):
        self.root_path = root_path or self._find_project_root()
        logger.info(f"Project root is {self.root_path}")

    def _find_project_root(self, marker_file: str = "README.MD", max_depth: int = 10) -> Path:
        """Find project root directory"""
        current_path = Path.cwd().resolve()
        for _ in range(max_depth):
            if (current_path / marker_file).exists():
                return current_path
            if current_path == current_path.parent:
                break
            current_path = current_path.parent
        raise FileNotFoundError(
            f"Root directory containing {marker_file} not found within {max_depth} levels"
        )

    def load_paths(self, directory: Path) -> List[str]:
        """Load all file paths from directory"""
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []
        return [str(p.resolve()) for p in directory.rglob("*") if p.is_file()]

    def get_datasets(self) -> DatasetPaths:
        """Get all dataset paths"""
        datasets = {}
        for split in ['train', 'valid', 'test']:
            images_dir = self.root_path / f'dataset/processed/{split}/images'
            masks_dir = self.root_path / f'dataset/processed/{split}/masks'
            
            images = sorted([
                img for img in self.load_paths(images_dir)
                if "_leftImg8bit.png" in img
            ])
            masks = sorted([
                msk for msk in self.load_paths(masks_dir)
                if "_labelIds.png" in msk
            ])
            
            self._check_paths(images, masks)
            
            datasets[f'{split}_images'] = images
            datasets[f'{split}_masks'] = masks
        
        return DatasetPaths(**datasets)

    @staticmethod
    def _check_paths(images: List[str], masks: List[str]) -> None:
        """Verify image and mask pairs match"""
        if len(images) != len(masks):
            raise ValueError(
                f"Number of images ({len(images)}) does not match masks ({len(masks)})"
            )
        
        for img_path, msk_path in zip(images, masks):
            img_name = Path(img_path).stem.split("_")[:3]
            msk_name = Path(msk_path).stem.split("_")[:3]
            if img_name != msk_name:
                raise ValueError(f"Mismatch: {img_path} and {msk_path}")


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


def find_project_root(current_path, marker_file="README.MD", max_depth=10):
    """
    Find the project's root directory by searching upwards until a marker file is found.

    Args:
        current_path (str or Path): The current path from which to start searching upwards.
        marker_file (str): The file used as a marker for the project root.
        max_depth (int): Maximum number of levels to search upwards.

    Raises:
        FileNotFoundError: If the marker file is not found within the maximum depth.

    Returns:
        Path: The Path object of the project root directory.
    """
    current_path = Path(current_path).resolve()
    depth = 0

    while current_path != current_path.root and depth < max_depth:
        print(f"Searching in: {current_path}")  # Debugging line
        if (current_path / marker_file).exists():
            print(f"Found marker file at: {current_path}")  # Debugging line
            return current_path
        current_path = current_path.parent
        depth += 1

    raise FileNotFoundError(f"Root directory containing {marker_file} not found within {max_depth} levels.")


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



def get_datasets(root_path=None):
    """
    Retrieve paths for train, validation, and test sets of images and masks.
    Ensure that image-mask pairs match and return a dictionary of datasets.

    Args:
        root_path (Path, optional): The root directory of the project. If None, 
                                    it will be determined using find_project_root.

    Returns:
        dict: A dictionary containing lists of file paths for 'train_images', 
              'train_masks', 'valid_images', 'valid_masks', 'test_images', and 'test_masks'.
    """
    if root_path is None:
        root_path = find_project_root(Path.cwd())  # Default to searching from cwd

    print(f"Project root is {root_path}")

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
    root_train_images = root_path / 'dataset/processed/train/images'
    train_images = load_paths(root_train_images)
    train_images = [img for img in train_images if "_leftImg8bit.png" in img]
    datasets_paths["train_images"] = sorted(train_images)

    root_train_masks = root_path / 'dataset/processed/train/masks'
    train_masks = load_paths(root_train_masks)
    train_masks = [msk for msk in train_masks if "_labelIds.png" in msk]
    datasets_paths["train_masks"] = sorted(train_masks)

    check_paths(datasets_paths["train_images"], datasets_paths["train_masks"])

    # Load validation images and masks
    root_valid_images = root_path / 'dataset/processed/valid/images'
    valid_images = load_paths(root_valid_images)
    valid_images = [img for img in valid_images if "_leftImg8bit.png" in img]
    datasets_paths["valid_images"] = sorted(valid_images)

    root_valid_masks = root_path / 'dataset/processed/valid/masks'
    valid_masks = load_paths(root_valid_masks)
    valid_masks = [msk for msk in valid_masks if "_labelIds.png" in msk]
    datasets_paths["valid_masks"] = sorted(valid_masks)

    check_paths(datasets_paths["valid_images"], datasets_paths["valid_masks"])

    # Load test images and masks
    root_test_images = root_path / 'dataset/processed/test/images'
    test_images = load_paths(root_test_images)
    test_images = [img for img in test_images if "_leftImg8bit.png" in img]
    datasets_paths["test_images"] = sorted(test_images)

    root_test_masks = root_path / 'dataset/processed/test/masks'
    test_masks = load_paths(root_test_masks)
    test_masks = [msk for msk in test_masks if "_labelIds.png" in msk]
    datasets_paths["test_masks"] = sorted(test_masks)

    check_paths(datasets_paths["test_images"], datasets_paths["test_masks"])

    return datasets_paths