import os
from pathlib import Path




def load_paths(directory):
    full_paths=[]
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.abspath(os.path.join(root, file))
            full_paths.append(full_path)
    return full_paths




def find_project_root(current_path, marker_file="README.md"):
    current_path = Path(current_path).resolve()
    while current_path != current_path.root:
        if (current_path / marker_file).exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError(f"Root directory containing {marker_file} not found.")


    

def check_paths(imgs, msks):
    
    for pair in zip(imgs, msks): 
        img_pth = pair[0].split("/")[-1].split("_")
        msk_pth = pair[0].split("/")[-1].split("_")
        assert img_pth == msk_pth, f"Paths of image {pair[0]} and mask {pair[1]} do not match"
    
    assert len(imgs) == len(msks), f"Number of images {len(imgs)} does not match the number of masks {len(msks)}"
    
    print("Paths are correct - check passed")    




def get_datasets():
    
    cwd = Path.cwd()
    project_root = find_project_root(cwd)
    print(f"project root is {project_root}")
    
    datasets_paths = {}
    mount_points = {}
    
    # List of dataset names
    dataset_names = [
        "segmentation_images_train",
        "segmentation_masks_train",
        "segmentation_images_valid",
        "segmentation_masks_valid",
        "segmentation_images_test",
        "segmentation_masks_test",
    ]
    # registered_valid_test_dataset_names = ["segmentation_images_valid_v2","segmentation_masks_valid_v2","segmentation_images_test_v2","segmentation_masks_test_v2"]

    root_train_images_path = project_root / 'data/datasets/train/images'
    train_images_paths = load_paths(root_train_images_path)
    train_images_paths = [img for img in train_images_paths if "_leftImg8bit.png" in img]
    datasets_paths[dataset_names[0]] = train_images_paths
    
    root_train_masks_path = project_root / 'data/datasets/train/masks'
    print(f"full path for training masks is {root_train_masks_path}")
    train_masks_paths = load_paths(root_train_masks_path)
    train_masks_paths = [msk for msk in train_masks_paths if "_labelIds.png" in msk]
    datasets_paths[dataset_names[1]] = train_masks_paths
    
    check_paths(datasets_paths[dataset_names[0]], datasets_paths[dataset_names[1]])

    root_valid_images_path = project_root / 'data/datasets/valid/images'
    valid_images_paths = load_paths(root_valid_images_path)
    valid_images_paths = [img for img in valid_images_paths if "_leftImg8bit.png" in img]
    datasets_paths[dataset_names[2]] = valid_images_paths
    
    root_valid_masks_path = project_root / 'data/datasets/valid/masks'
    valid_masks_paths = load_paths(root_valid_masks_path)
    valid_masks_paths = [msk for msk in valid_masks_paths if "_labelIds.png" in msk]
    datasets_paths[dataset_names[3]] = valid_masks_paths

    check_paths(datasets_paths[dataset_names[2]], datasets_paths[dataset_names[3]])
    
    root_test_images_path = project_root / 'data/datasets/test/images'
    test_images_paths = load_paths(root_test_images_path)
    test_images_paths = [msk for msk in test_images_paths if "_leftImg8bit.png" in msk]
    datasets_paths[dataset_names[4]] = test_images_paths

    root_test_masks_path = project_root / 'data/datasets/test/masks'
    test_masks_paths = load_paths(root_test_masks_path)
    test_masks_paths = [msk for msk in test_masks_paths if "_labelIds.png" in msk]
    datasets_paths[dataset_names[5]] = test_masks_paths

    check_paths(datasets_paths[dataset_names[4]], datasets_paths[dataset_names[5]])

    return datasets_paths
