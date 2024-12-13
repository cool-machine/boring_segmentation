"""
process_data.py

This module provides functions for creating TensorFlow Datasets from loaded paths, 
including reading images and labels, normalization, label mapping, augmentation, 
and resizing for model input requirements.
"""

import tensorflow as tf
from tensorflow.keras import backend as K

# Adjust this import to match your project structure. 
# If load_data.py is in the same folder, you can use:
from .load_data import get_datasets


def read_image(file_path):
    """
    Read an image file and decode it into a TensorFlow tensor.

    Args:
        file_path (str): The file path of the image.

    Returns:
        tf.Tensor: The decoded and float32 converted image of shape [1024, 2048, 3].
    """
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape([1024, 2048, 3])
    return image


def read_label(file_path):
    """
    Read a label file and decode it into a TensorFlow tensor.

    Args:
        file_path (str): The file path of the mask/label.

    Returns:
        tf.Tensor: The decoded label of shape [1024, 2048, 1].
    """
    label = tf.io.read_file(file_path)
    label = tf.image.decode_image(label, channels=1)
    label = tf.image.convert_image_dtype(label, tf.uint8)
    label.set_shape([1024, 2048, 1])
    return label


def normalize(input_image, input_mask):
    """
    Normalize the input image using predefined mean and std values, and clip it to [0,1].
    Labels are left unchanged.

    Args:
        input_image (tf.Tensor): The input image tensor of shape [H, W, 3].
        input_mask (tf.Tensor): The corresponding label tensor.

    Returns:
        tuple: (normalized_image, unchanged_label)
    """
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_image = (input_image - mean) / tf.maximum(std, K.epsilon())
    input_image = tf.clip_by_value(input_image, 0.0, 1.0)

    return input_image, input_mask


def retrieve_mask_mappings():
    """
    Retrieve original classes, a class mapping dictionary, 
    and new labels for a reduced set of classes.

    Returns:
        tuple: (original_classes, class_mapping, new_labels)
    """
    original_classes = [
        'road', 'sidewalk', 'parking', 'rail track', 'person', 'rider', 'car', 'truck',
        'bus', 'on rails', 'motorcycle', 'bicycle', 'caravan', 'trailer', 'building', 
        'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'pole group', 
        'traffic sign', 'traffic light', 'vegetation', 'terrain', 'sky', 'ground', 
        'dynamic', 'static'
    ]

    class_mapping = {
        'road': 'flat', 'sidewalk': 'flat',
        'parking': 'flat', 'rail track': 'flat',
        'person': 'human', 'rider': 'human',
        'car': 'vehicle', 'truck': 'vehicle',
        'bus': 'vehicle', 'on rails': 'vehicle',
        'motorcycle': 'vehicle', 'bicycle': 'vehicle',
        'caravan': 'vehicle', 'trailer': 'vehicle',
        'building': 'construction', 'wall': 'construction',
        'fence': 'construction', 'guard rail': 'construction',
        'bridge': 'construction', 'tunnel': 'construction',
        'pole': 'object', 'pole group': 'object',
        'traffic sign': 'object', 'traffic light': 'object',
        'vegetation': 'nature', 'terrain': 'nature',
        'sky': 'sky', 'ground': 'void',
        'dynamic': 'void', 'static': 'void'
    }

    new_labels = {
        'flat': 0, 'human': 1,
        'vehicle': 2, 'construction': 3,
        'object': 4, 'nature': 5,
        'sky': 6, 'void': 7
    }

    return original_classes, class_mapping, new_labels


def map_labels_tf(label_image, original_classes, class_mapping, new_labels):
    """
    Map the original 30-class labels to a reduced set of 8 classes.

    Args:
        label_image (tf.Tensor): The input label image tensor [H, W, 1].
        original_classes (list of str): The full set of original classes.
        class_mapping (dict): Maps each original class to one of the reduced classes.
        new_labels (dict): Maps the reduced classes to numeric labels.

    Returns:
        tf.Tensor: The mapped label image.
    """
    label_image = tf.squeeze(label_image)
    label_shape = tf.shape(label_image)
    mapped_label_image = tf.zeros_like(label_image, dtype=tf.uint8)

    for original_class, new_class in class_mapping.items():
        original_class_index = tf.cast(original_classes.index(original_class), tf.uint8)
        new_class_index = tf.cast(new_labels[new_class], tf.uint8)

        mask = tf.equal(tf.cast(label_image, tf.int32), tf.cast(original_class_index, tf.int32))
        fill_val = tf.fill(label_shape, new_class_index)
        mapped_label_image = tf.where(mask, fill_val, mapped_label_image)

    label = tf.expand_dims(mapped_label_image, axis=-1)
    label = tf.image.convert_image_dtype(label, tf.uint8)
    return label


def augment_image_and_label(image, label, augment_prob=0.3):
    """
    Augment image and label with random flips and brightness/contrast changes.

    Args:
        image (tf.Tensor): The input image tensor [H, W, 3].
        label (tf.Tensor): The input label tensor [H, W, 1].
        augment_prob (float): Probability for performing augmentations.

    Returns:
        tuple: (augmented_image, augmented_label)
    """
    def augment():
        seed = tf.random.uniform([2], maxval=64, dtype=tf.int32)

        # Horizontal flip
        if tf.random.uniform([], 0, 1.0) > augment_prob:
            image_in = tf.image.stateless_random_flip_left_right(image, seed=seed)
            label_in = tf.image.stateless_random_flip_left_right(label, seed=seed)
        else:
            image_in = image
            label_in = label

        # Vertical flip
        seed = tf.random.uniform([2], maxval=64, dtype=tf.int32)
        if tf.random.uniform([], 0, 1.0) > augment_prob:
            image_in = tf.image.stateless_random_flip_up_down(image_in, seed=seed)
            label_in = tf.image.stateless_random_flip_up_down(label_in, seed=seed)

        # Brightness and contrast
        seed = tf.random.uniform([2], maxval=64, dtype=tf.int32)
        if tf.random.uniform([], 0, 1.0) > augment_prob:
            image_in = tf.image.stateless_random_brightness(image_in, max_delta=0.2, seed=seed)
            image_in = tf.image.stateless_random_contrast(image_in, lower=0.8, upper=1.2, seed=seed)

        return image_in, label_in

    def no_augment():
        return image, label

    random_number = tf.random.uniform([], 0, 1.0)
    image, label = tf.cond(random_number < augment_prob, augment, no_augment)

    image = tf.clip_by_value(image, 0.0, 1.0)
    label = tf.image.convert_image_dtype(label, tf.uint8)

    return image, label


def load_dataset_unet(batch_size=1):
    """
    Create train and validation tf.data.Datasets suitable for U-Net training.

    Args:
        batch_size (int): Batch size for the dataset.

    Returns:
        tuple: (dataset_train, dataset_valid) as tf.data.Dataset objects
    """
    datasets = get_datasets()

    train_image_paths = datasets["train_images"]
    train_mask_paths = datasets["train_masks"]
    val_image_paths = datasets["valid_images"]
    val_mask_paths = datasets["valid_masks"]

    dataset_train = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mask_paths))
    dataset_valid = tf.data.Dataset.from_tensor_slices((val_image_paths, val_mask_paths))

    # Read and decode images and labels
    dataset_train = dataset_train.map(lambda img, msk: (read_image(img), read_label(msk)), num_parallel_calls=tf.data.AUTOTUNE)
    dataset_valid = dataset_valid.map(lambda img, msk: (read_image(img), read_label(msk)), num_parallel_calls=tf.data.AUTOTUNE)

    # Map labels to reduced classes
    original_classes, class_mapping, new_labels = retrieve_mask_mappings()
    dataset_train = dataset_train.map(lambda im, m: (im, map_labels_tf(m, original_classes, class_mapping, new_labels)), 
                                      num_parallel_calls=tf.data.AUTOTUNE)
    dataset_valid = dataset_valid.map(lambda im, m: (im, map_labels_tf(m, original_classes, class_mapping, new_labels)), 
                                      num_parallel_calls=tf.data.AUTOTUNE)

    # Normalize and augment
    dataset_train = dataset_train.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_valid = dataset_valid.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_train = dataset_train.map(augment_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    dataset_train = dataset_train.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    dataset_valid = dataset_valid.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return dataset_train, dataset_valid


def resize_images(image, label):
    """
    Resize the image and label to the required input shapes for specific models.

    Args:
        image (tf.Tensor): The input image tensor [H, W, 3].
        label (tf.Tensor): The input label tensor [H, W, 1].

    Returns:
        tuple: (resized_image, resized_label)
    """
    image = tf.image.resize(image, size=[512, 1024], method=tf.image.ResizeMethod.BILINEAR)
    label = tf.image.resize(label, size=[128, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = tf.image.convert_image_dtype(label, tf.uint8)

    # Transpose to [C, H, W] format if required
    image = tf.transpose(image, perm=[2, 0, 1])
    output_mask = tf.transpose(label, perm=[2, 0, 1])

    output_image = tf.image.convert_image_dtype(image, tf.float32)
    output_image = tf.clip_by_value(output_image, 0.0, 1.0)
    
    return output_image, output_mask


def load_dataset_segf(batch_size=2):
    """
    Create a dataset for a Segformer-like model, resizing and normalizing images and labels.

    Args:
        batch_size (int): Batch size for the dataset.

    Returns:
        tf.data.Dataset: The training dataset.
    """
    datasets = get_datasets()
    # These keys should match your dataset naming if you plan to adapt this function
    train_image_paths = datasets["train_images"]
    train_mask_paths = datasets["train_masks"]
    val_image_paths = datasets["valid_images"]
    val_mask_paths = datasets["valid_masks"]

    dataset_train = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mask_paths))
    dataset_valid = tf.data.Dataset.from_tensor_slices((val_image_paths, val_mask_paths))

    # Retrieve mappings
    original_classes, class_mapping, new_labels = retrieve_mask_mappings()

    # Map labels
    dataset_train = dataset_train.map(lambda img, msk: (img, map_labels_tf(msk, original_classes, class_mapping, new_labels)),
                                      num_parallel_calls=tf.data.AUTOTUNE)
    dataset_valid = dataset_valid.map(lambda img, msk: (img, map_labels_tf(msk, original_classes, class_mapping, new_labels)),
                                      num_parallel_calls=tf.data.AUTOTUNE)

    # Normalize, augment, and resize
    dataset_train = dataset_train.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_valid = dataset_valid.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    dataset_train = dataset_train.map(augment_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_train = dataset_train.map(resize_images, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_valid = dataset_valid.map(resize_images, num_parallel_calls=tf.data.AUTOTUNE)

    dataset_train = dataset_train.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    dataset_valid = dataset_valid.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return dataset_train, dataset_valid



# import tensorflow as tf
# from src.data_processing.data_loading import get_datasets
# from src.data_processing.utils import load_paths

# Function to read and process an image file
# def read_image(file_path):
#     """
#     Reads and decodes an image file into a TensorFlow tensor.

#     Args:
#         file_path (str): Path to the image file.

#     Returns:
#         tf.Tensor: A 3D tensor representing the image with shape [1024, 2048, 3] 
#                    and pixel values normalized to [0, 1].
#     """
#     image = tf.io.read_file(file_path)  # Read the image file as raw bytes
#     image = tf.image.decode_image(image, channels=3)  # Decode the image, assuming 3 channels (RGB)
#     image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32 with values in [0, 1]
#     image.set_shape([1024, 2048, 3])  # Explicitly set the expected image shape
#     return image

# # Function to read and process a label/mask file
# def read_label(file_path):
#     """
#     Reads and decodes a label/mask file into a TensorFlow tensor.

#     Args:
#         file_path (str): Path to the label/mask file.

#     Returns:
#         tf.Tensor: A 3D tensor representing the label with shape [1024, 2048, 1] 
#                    and pixel values as integers.
#     """
#     label = tf.io.read_file(file_path)  # Read the label file as raw bytes
#     label = tf.image.decode_image(label, channels=1)  # Decode the label, assuming 1 channel (grayscale)
#     label = tf.image.convert_image_dtype(label, tf.uint8)  # Convert to uint8 with integer values
#     label.set_shape([1024, 2048, 1])  # Explicitly set the expected label shape
#     return label

# # Function to preprocess an image and its corresponding label/mask
# def preprocess_image_and_label(image, label):
#     """
#     Normalizes the image and clips pixel values to [0, 1]. Labels are returned unchanged.

#     Args:
#         image (tf.Tensor): A 3D tensor representing the image.
#         label (tf.Tensor): A 3D tensor representing the label/mask.

#     Returns:
#         Tuple[tf.Tensor, tf.Tensor]: The normalized image and the unchanged label.
#     """
#     # Mean and standard deviation values for normalization (specific to the model's requirements)
#     mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)  # Per-channel mean values
#     std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)  # Per-channel standard deviation values
    
#     # Normalize the image: subtract the mean and divide by the standard deviation
#     image = (image - mean) / tf.maximum(std, tf.keras.backend.epsilon())
#     # Clip pixel values to the range [0, 1] for stability
#     image = tf.clip_by_value(image, 0.0, 1.0)
    
#     return image, label  # Return the preprocessed image and the original label

# # Function to create and load TensorFlow datasets for training and validation
# def load_dataset(batch_size=1):
#     """
#     Creates TensorFlow datasets for training and validation, including preprocessing.

#     Args:
#         batch_size (int): The batch size to use for the datasets.

#     Returns:
#         Tuple[tf.data.Dataset, tf.data.Dataset]: The training and validation datasets.
#     """
#     # Retrieve file paths for images and masks using the data loading utility
#     datasets = get_datasets()
#     train_image_paths, train_mask_paths = datasets["train_images"], datasets["train_masks"]
#     val_image_paths, val_mask_paths = datasets["valid_images"], datasets["valid_masks"]

#     # Create TensorFlow datasets from the image and mask file paths
#     train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mask_paths))
#     val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_mask_paths))

#     # Map each image and mask file path to its decoded tensor
#     train_dataset = train_dataset.map(lambda img, lbl: (read_image(img), read_label(lbl)))
#     val_dataset = val_dataset.map(lambda img, lbl: (read_image(img), read_label(lbl)))

#     # Preprocess the image and label, batch the data, and enable prefetching for performance
#     train_dataset = (
#         train_dataset
#         .map(preprocess_image_and_label)  # Normalize images and prepare labels
#         .batch(batch_size)               # Batch the dataset
#         .prefetch(tf.data.AUTOTUNE)      # Prefetch to improve input pipeline performance
#     )
#     val_dataset = (
#         val_dataset
#         .map(preprocess_image_and_label)  # Normalize images and prepare labels
#         .batch(batch_size)                # Batch the dataset
#         .prefetch(tf.data.AUTOTUNE)       # Prefetch to improve input pipeline performance
#     )

#     return train_dataset, val_dataset  # Return the training and validation datasets





# import tensorflow as tf
# from data.data_loading import get_datasets
# from data.utils import load_paths

# def read_image(file_path):
#     image = tf.io.read_file(file_path)
#     image = tf.image.decode_image(image, channels=3)
#     image = tf.image.convert_image_dtype(image, tf.float32)
#     image.set_shape([1024, 2048, 3]) 
#     return image

# def read_label(file_path):
#     label = tf.io.read_file(file_path)
#     label = tf.image.decode_image(label, channels=1)
#     label = tf.image.convert_image_dtype(label, tf.uint8)
#     label.set_shape([1024, 2048, 1])
#     return label

# def preprocess_image_and_label(image, label):
#     # Normalize and clip image
#     mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
#     std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
#     image = (image - mean) / tf.maximum(std, tf.keras.backend.epsilon())
#     image = tf.clip_by_value(image, 0.0, 1.0)
#     return image, label

# def load_dataset(batch_size=1):
#     datasets = get_datasets()
#     train_image_paths, train_mask_paths = datasets["train_images"], datasets["train_masks"]
#     val_image_paths, val_mask_paths = datasets["valid_images"], datasets["valid_masks"]

#     train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mask_paths))
#     val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_mask_paths))

#     train_dataset = train_dataset.map(lambda img, lbl: (read_image(img), read_label(lbl)))
#     val_dataset = val_dataset.map(lambda img, lbl: (read_image(img), read_label(lbl)))

#     train_dataset = train_dataset.map(preprocess_image_and_label).batch(batch_size).prefetch(tf.data.AUTOTUNE)
#     val_dataset = val_dataset.map(preprocess_image_and_label).batch(batch_size).prefetch(tf.data.AUTOTUNE)

#     return train_dataset, val_dataset

# import tensorflow as tf
# from tensorflow.keras import backend as K
# # from data.data_processing.load_data_paths import get_datasets
# from src.data_processing.load_data import get_datasets


# def read_image(file_path):
#     image = tf.io.read_file(file_path)
#     image = tf.image.decode_image(image, channels=3)
#     image = tf.image.convert_image_dtype(image, tf.float32)
#     image.set_shape([1024, 2048, 3]) 
#     return image


# def read_label(file_path):
#     label = tf.io.read_file(file_path)
#     label = tf.image.decode_image(label, channels=1)
#     label = tf.image.convert_image_dtype(label, tf.uint8)
#     label.set_shape([1024, 2048, 1])
#     return label


# def normalize(input_image, input_mask):
#     '''
#     Function normalize adjusts values of images to the values that 
#     are expected by trained models (in this case Segformer). 
#     It uses means and standard deviations 
#     for each channel of image. 

#     Remarque: labels do not need adjustment

#     Arguments: 
#         input_image: color image received by the function
#         input_mask: mask that corresponds to the image
    
#     '''

#     # Hard-code values of mean and standard deviation 
#     # required to adjust images for segformer
#     mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
#     std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

#     # Adjust pixel values for input images
#     input_image = tf.image.convert_image_dtype(input_image, tf.float32)
#     input_image = (input_image - mean) / tf.maximum(std, K.epsilon())

#     # Clip the pixel values of images to ensure they are within [0, 1]
#     input_image = tf.clip_by_value(input_image, clip_value_min=0.0, clip_value_max=1.0)
    
#     return input_image, input_mask


# def retrieve_mask_mappings():
    
#     # Original 30 classes
#     original_classes = [
#         'road', 'sidewalk', 'parking', 'rail track', 'person', 'rider', 'car', 'truck', 'bus', 'on rails',
#         'motorcycle', 'bicycle', 'caravan', 'trailer', 'building', 'wall', 'fence', 'guard rail', 'bridge',
#         'tunnel', 'pole', 'pole group', 'traffic sign', 'traffic light', 'vegetation', 'terrain', 'sky',
#         'ground', 'dynamic', 'static'
#     ]

#     # Mapping to 8 major groups
#     class_mapping = {
#         'road': 'flat', 'sidewalk': 'flat', 
#         'parking': 'flat', 'rail track': 'flat',
#         'person': 'human', 'rider': 'human',
#         'car': 'vehicle', 'truck': 'vehicle', 
#         'bus': 'vehicle', 'on rails': 'vehicle',
#         'motorcycle': 'vehicle', 'bicycle': 'vehicle', 
#         'caravan': 'vehicle', 'trailer': 'vehicle',
#         'building': 'construction', 'wall': 'construction', 
#         'fence': 'construction', 'guard rail': 'construction',
#         'bridge': 'construction', 'tunnel': 'construction',
#         'pole': 'object', 'pole group': 'object', 
#         'traffic sign': 'object', 'traffic light': 'object',
#         'vegetation': 'nature', 'terrain': 'nature',
#         'sky': 'sky', 'ground': 'void', 
#         'dynamic': 'void', 'static': 'void'
#     }

#     # New labels for the 8 major groups
#     new_labels = {
#         'flat': 0, 'human': 1, 
#         'vehicle': 2, 'construction': 3, 
#         'object': 4, 'nature': 5, 
#         'sky': 6, 'void': 7
#     }

#     return original_classes, class_mapping, new_labels


# def map_labels_tf(label_image, original_classes, class_mapping, new_labels):
#     label_image = tf.squeeze(label_image)
#     label_image_shape = tf.shape(label_image)
#     mapped_label_image = tf.zeros_like(label_image, dtype=tf.uint8)
#     for original_class, new_class in class_mapping.items():
#         original_class_index = tf.cast(original_classes.index(original_class), tf.uint8)
#         new_class_index = tf.cast(new_labels[new_class], tf.uint8)
#         mask = tf.equal(tf.cast(label_image,tf.int32), tf.cast(original_class_index, tf.int32))
#         fill_val = tf.fill(label_image_shape, tf.cast(new_class_index, tf.uint8))  
#         mapped_label_image = tf.where(mask, fill_val, mapped_label_image)
#     label = tf.expand_dims(mapped_label_image, axis=-1)  # Add back the last dimension
#     label = tf.image.convert_image_dtype(label, tf.uint8)
#     return label


# def augment_image_and_label(image, label, augment_prob=0.3):
    
#     def augment():
#         # Generate a random seed
#         seed = tf.random.uniform([2], maxval=64, dtype=tf.int32)
#         # Apply augmentations with the seed
#         if tf.random.uniform([], minval=0, maxval=1.0, dtype=tf.float32) > augment_prob:
#             image_in = tf.image.stateless_random_flip_left_right(image, seed=seed)
#             label_in = tf.image.stateless_random_flip_left_right(label, seed=seed)
#         else:
#             image_in = image
#             label_in = label

#         seed = tf.random.uniform([2], maxval=64, dtype=tf.int32)
        
#         if tf.random.uniform([], minval=0, maxval=1.0, dtype=tf.float32) > augment_prob:
#             image_in = tf.image.stateless_random_flip_up_down(image_in, seed=seed)
#             label_in = tf.image.stateless_random_flip_up_down(label_in, seed=seed)

#         if tf.random.uniform([], minval=0, maxval=1.0, dtype=tf.float32) > augment_prob:
#             image_in = tf.image.stateless_random_brightness(image_in, max_delta=0.2, seed=seed)
#             image_in = tf.image.stateless_random_contrast(image_in, lower=0.8, upper=1.2, seed=seed)

#         return image_in, label_in

#     def no_augment():
#         return image, label

#     # Use a stateless random number for consistency
#     random_number = tf.random.uniform([], minval=0, maxval=1.0, dtype=tf.float32)
    
#     image, label = tf.cond(random_number < augment_prob, augment, no_augment)

#     image = tf.image.convert_image_dtype(image, tf.float32)

#     # Clip the values to ensure they are within [0, 1]
#     image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

#     # Make sure labels are of tf.uint8 format
#     label = tf.image.convert_image_dtype(label, tf.uint8)
    
#     return image, label


# def load_dataset_unet(batch_size=1):
    
#     # Get full paths for images and masks
#     datasets = get_datasets()
#         # print(f"print training dataset shape{}")

#     # for key in datasets:
#     #     print(key)
#     #     print(len(datasets[key]))
#     #     print(datasets[key][0])
#     #     print()

    
#     train_image_paths = datasets["train_images"]
#     train_mask_paths = datasets["train_masks"]
    
#     val_image_paths = datasets["valid_images"]
#     val_mask_paths = datasets["valid_masks"]
    
#     # Create a dataset from the tuples of (image_path, mask_path)
#     dataset_train = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mask_paths))
#     dataset_valid = tf.data.Dataset.from_tensor_slices((val_image_paths, val_mask_paths))

#     # Read image paths in tensorflow
#     dataset_train = dataset_train.map(lambda image, mask: (read_image(image), read_label(mask)))
#     dataset_valid = dataset_valid.map(lambda image, mask: (read_image(image), read_label(mask)))

#     # Group 30 categories of labels into eight categories
#     original_classes, class_mapping, new_labels = retrieve_mask_mappings()
    
#     dataset_train = dataset_train.map(lambda image, mask: (image, map_labels_tf(mask, 
#                                                                     original_classes, 
#                                                                     class_mapping, 
#                                                                     new_labels)),           
#                                                            num_parallel_calls=tf.data.AUTOTUNE)

#     dataset_valid = dataset_valid.map(lambda image, mask: (image, map_labels_tf(mask, 
#                                                                     original_classes, 
#                                                                     class_mapping, 
#                                                                     new_labels)),           
#                                                            num_parallel_calls=tf.data.AUTOTUNE)    

#     # Normalize examples
#     dataset_train = dataset_train.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     dataset_valid = dataset_valid.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

#     dataset_train = dataset_train.map(augment_image_and_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
#     dataset_train = dataset_train.batch(batch_size, drop_remainder=True)
#     dataset_valid= dataset_valid.batch(batch_size, drop_remainder=True)

#     dataset_train = dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#     dataset_valid = dataset_valid.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
#     return dataset_train, dataset_valid
    

# def resize_images(image, label):

#     image = tf.image.resize(image, size=[512, 1024], method=tf.image.ResizeMethod.BILINEAR)

#     label = tf.image.resize(label, size=[128, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#     label = tf.image.convert_image_dtype(label, tf.uint8)

#     image = tf.transpose(image, perm=[2, 0, 1])
#     output_mask = tf.transpose(label, perm=[2, 0, 1])

#     output_image = tf.image.convert_image_dtype(image, tf.float32)
#     output_image = tf.clip_by_value(output_image, clip_value_min=0.0, clip_value_max=1.0)
    
#     return output_image, output_mask


# def load_dataset_segf(batch_size=2):

#     # Get full paths for images and masks
#     datasets = get_datasets()
    
#     train_image_paths = datasets["segmentation_images_train"]
#     train_mask_paths = datasets["segmentation_masks_train"]
    
#     val_image_paths = datasets["segmentation_images_valid"]
#     val_mask_paths = datasets["segmentation_masks_valid"]

#     # Create a dataset from the tuples of (image_path, mask_path)
#     dataset_train = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mask_paths))
#     dataset_valid = tf.data.Dataset.from_tensor_slices((val_image_paths, val_mask_paths))

#     # Group 30 categories of labels into eight categories
#     original_classes, class_mapping, new_labels = retrieve_mask_mappings()
    
#     dataset_train = dataset_train.map(lambda image, mask: (image, map_labels_tf(mask, 
#                                                                     original_classes, 
#                                                                     class_mapping, 
#                                                                     new_labels)),           
#                                                            num_parallel_calls=tf.data.AUTOTUNE)

#     dataset_valid = dataset_valid.map(lambda image, mask: (image, map_labels_tf(mask, 
#                                                                     original_classes, 
#                                                                     class_mapping, 
#                                                                     new_labels)),           
#                                                            num_parallel_calls=tf.data.AUTOTUNE)

#     # Normalize examples
#     dataset_train = dataset_train.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     dataset_valid = dataset_valid.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

#     dataset_train = dataset_train.map(augment_image_and_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)

#     dataset_train = dataset_train.map(resize_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     dataset_valid = dataset_valid.map(resize_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

#     dataset_train = dataset_train.batch(batch_size, drop_remainder=True)
#     dataset_valid= dataset_valid.batch(batch_size, drop_remainder=True)

#     dataset_train = dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#     dataset_valid = dataset_valid.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

#     return dataset