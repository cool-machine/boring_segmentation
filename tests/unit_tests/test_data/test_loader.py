import os
import pytest
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.data.loader import PathLoader, DatasetPaths, get_datasets


def test_find_project_root(tmp_path, monkeypatch):
    # Create a marker file (e.g., README.MD) in the tmp_path
    marker = tmp_path / "README.MD"
    marker.write_text("Project marker")
    
    # Create a subdirectory and set it as the current working directory.
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)
    
    # _find_project_root() should now find the tmp_path as the project root.
    loader = PathLoader()
    assert loader.root_path == tmp_path.resolve()

def test_load_paths(tmp_path):
    # Create a temporary directory with two files.
    test_dir = tmp_path / "test_files"
    test_dir.mkdir()
    file1 = test_dir / "file1.txt"
    file1.write_text("hello")
    file2 = test_dir / "file2.txt"
    file2.write_text("world")
    
    loader = PathLoader(root_path=tmp_path)
    paths = loader.load_paths(test_dir)
    # Ensure both files are found.
    assert len(paths) == 2
    assert str(file1.resolve()) in paths
    assert str(file2.resolve()) in paths

def test_check_paths_valid():
    # Provide valid image and mask file names that share the same first three parts.
    images = [
        "city1_region1_seq1_leftImg8bit.png",
        "city2_region2_seq2_leftImg8bit.png"
    ]
    masks = [
        "city1_region1_seq1_labelIds.png",
        "city2_region2_seq2_labelIds.png"
    ]
    # Should not raise an exception.
    PathLoader._check_paths(images, masks)

def test_check_paths_invalid():
    images = ["city1_region1_seq1_leftImg8bit.png"]
    masks = ["city1_regionX_seq1_labelIds.png"]  # Mismatch: 'region1' vs 'regionX'
    with pytest.raises(ValueError):
        PathLoader._check_paths(images, masks)

def test_get_datasets(tmp_path):
    # Create a fake dataset directory structure:
    #   tmp_path/dataset/processed/{split}/images and .../masks for each split.
    for split in ['train', 'valid', 'test']:
        images_dir = tmp_path / f"dataset/processed/{split}/images"
        masks_dir = tmp_path / f"dataset/processed/{split}/masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Create two dummy files per split with matching names.
        for i in range(2):
            base_name = f"city{i}_region{i}_seq{i}"
            img_file = images_dir / f"{base_name}_leftImg8bit.png"
            msk_file = masks_dir / f"{base_name}_labelIds.png"
            img_file.write_text("image content")
            msk_file.write_text("mask content")
    
    loader = PathLoader(root_path=tmp_path)
    datasets = loader.get_datasets()
    
    # Verify that each split contains two images and two masks.
    assert len(datasets.train_images) == 2
    assert len(datasets.train_masks) == 2
    assert len(datasets.valid_images) == 2
    assert len(datasets.valid_masks) == 2
    assert len(datasets.test_images) == 2
    assert len(datasets.test_masks) == 2

def test_get_datasets_legacy(tmp_path):
    # Create a minimal fake dataset structure for each split.
    for split in ['train', 'valid', 'test']:
        images_dir = tmp_path / f"dataset/processed/{split}/images"
        masks_dir = tmp_path / f"dataset/processed/{split}/masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = "city0_region0_seq0"
        img_file = images_dir / f"{base_name}_leftImg8bit.png"
        msk_file = masks_dir / f"{base_name}_labelIds.png"
        img_file.write_text("image content")
        msk_file.write_text("mask content")
    
    datasets_dict = get_datasets(root_path=tmp_path)
    expected_keys = {
        "train_images", "train_masks",
        "valid_images", "valid_masks",
        "test_images", "test_masks"
    }
    assert set(datasets_dict.keys()) == expected_keys
