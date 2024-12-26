import os
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data_processing.load_data import (
    load_paths,
    find_project_root,
    check_paths,
    get_datasets,
)


@pytest.fixture
def project_structure():
    """
    Create a temporary project structure with a README.MD and a dataset folder.
    Returns the path to the created project root.
    """
    with TemporaryDirectory() as temp_dir:
        root_path = Path(temp_dir)
        print(f"Temporary root created at: {root_path}")  # Debugging line

        # Create marker file
        (root_path / "README.MD").touch()
        print(f"Created marker file: {root_path / 'README.MD'}")  # Debugging line

        # Create dataset directories
        for split in ["train", "valid", "test"]:
            images_dir = root_path / "dataset" / split / "images"
            masks_dir = root_path / "dataset" / split / "masks"
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)

            # Create mock files
            for i in range(2):
                base_name = f"city_{i:06}_{i:06}"
                (images_dir / f"{base_name}_leftImg8bit.png").touch()
                (masks_dir / f"{base_name}_labelIds.png").touch()

        print(f"Contents of project root: {list(root_path.iterdir())}")  # Debugging line
        yield root_path


def test_load_paths(project_structure):
    """
    Test that load_paths returns all files in a directory recursively.
    """
    images_dir = project_structure / "dataset" / "train" / "images"
    paths = load_paths(images_dir)
    # Expecting two image files
    assert len(paths) == 2, "Should find exactly two image files."
    assert all("_leftImg8bit.png" in p for p in paths), "All returned paths should be images."


def test_check_paths_success():
    """
    Test check_paths with matching image and mask lists.
    """
    images = [
        "/some/path/city_000000_000000_leftImg8bit.png",
        "/some/path/city_000001_000001_leftImg8bit.png"
    ]
    masks = [
        "/some/path/city_000000_000000_labelIds.png",
        "/some/path/city_000001_000001_labelIds.png"
    ]

    # Should not raise an assertion error
    check_paths(images, masks)


def test_check_paths_failure():
    """
    Test check_paths with non-matching image and mask lists
    to ensure it raises AssertionError.
    """
    images = [
        "/some/path/city_000000_000000_leftImg8bit.png"
    ]
    masks = [
        "/some/path/city_000001_000001_labelIds.png"
    ]

    with pytest.raises(AssertionError):
        check_paths(images, masks)


def test_get_datasets(project_structure, monkeypatch):
    """
    Test the get_datasets function to ensure it returns the correct dictionary structure.
    Use a custom project root from the fixture.
    """
    # Pass project_structure as the root_path to get_datasets
    datasets = get_datasets(root_path=project_structure)

    expected_keys = [
        "train_images",
        "train_masks",
        "valid_images",
        "valid_masks",
        "test_images",
        "test_masks",
    ]
    for key in expected_keys:
        assert key in datasets, f"{key} should be in the returned dataset dictionary."

    # Each should have two files
    assert len(datasets["train_images"]) == 2, "Should have two training images."
    assert len(datasets["train_masks"]) == 2, "Should have two training masks."
    assert len(datasets["valid_images"]) == 2, "Should have two validation images."
    assert len(datasets["valid_masks"]) == 2, "Should have two validation masks."
    assert len(datasets["test_images"]) == 2, "Should have two test images."
    assert len(datasets["test_masks"]) == 2, "Should have two test masks."


@pytest.mark.timeout(5)  # Fail the test if it takes longer than 5 seconds
def test_find_project_root_success(project_structure):
    """
    Test that find_project_root returns the correct project root
    when the marker file is present.
    """
    # Debugging: Print project structure
    print(f"Temporary project root: {project_structure}")
    print(f"Contents of project root: {list(project_structure.iterdir())}")

    # Ensure README.MD exists
    marker_file = project_structure / "README.MD"
    assert marker_file.exists(), f"Marker file {marker_file} does not exist."

    # Start searching from a subdirectory and ensure it finds the root
    start_path = project_structure / "dataset" / "train" / "images"
    found_root = find_project_root(start_path)

    assert found_root == project_structure, f"Expected {project_structure}, but found {found_root}."


def test_find_project_root_failure():
    """
    Test that find_project_root raises FileNotFoundError
    when the marker file is not present.
    """
    with TemporaryDirectory() as temp_dir:
        # No README.MD here
        start_path = Path(temp_dir)
        with pytest.raises(FileNotFoundError):
            find_project_root(start_path)
