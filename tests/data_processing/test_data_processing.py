# tests/data_processing/test_data_processing.py

# import pytest
# from data.data_processing import read_image, read_label

# def test_read_image():
#     image_path = "path/to/sample_image.png"
#     image = read_image(image_path)
#     assert image.shape == (1024, 2048, 3), "Image dimensions incorrect"

# def test_read_label():
#     label_path = "path/to/sample_label.png"
#     label = read_label(label_path)
#     assert label.shape == (1024, 2048, 1), "Label dimensions incorrect"


# import pytest
# import tensorflow as tf
# import numpy as np
# from data_processing.data_processing import read_image, read_label, normalize, resize_images
# from data_processing.data_processing import retrieve_mask_mappings, map_labels_tf, augment_image_and_label
# # from data.data_processing.load_data_paths import 

# # tf.random.uniform((1024, 2048, 1), minval=0, maxval=8, dtype=tf.int32)

# @pytest.fixture
# def image_label():
#     dummy_image = tf.random.uniform((1024, 2048, 3), minval=0, maxval=1, dtype=tf.float32) 
#     #np.random.randint(0, 255, (1024, 2048, 3), dtype=np.uint8)
#     dummy_label = tf.random.uniform((1024, 2048, 1), minval=0, maxval=30, dtype=tf.int32)
#     #np.random.randint(0, 8, (1024, 2048, 1), dtype=np.uint8)
#     return dummy_image, dummy_label

    
# # read_image
# def test_read_image(image_label):
#     # Create a dummy image file
#     dummy_image, _ = image_label #np.random.randint(0, 255, (1024, 2048, 3), dtype=np.uint8)
#     dummy_image_path = 'tests/dummy_image.png'
#     tf.keras.preprocessing.image.save_img(dummy_image_path, dummy_image)

#     # Read the image using read_image function
#     image = read_image(dummy_image_path)

#     # Check that the image has the correct shape and dtype
#     assert image.shape == (1024, 2048, 3), "Image shape is incorrect."
#     assert image.dtype == tf.float32, "Image dtype is not tf.float32."

#     # Clean up: Remove the dummy image file
#     import os
#     os.remove(dummy_image_path)


# # read_label
# def test_read_label(image_label):
#     # Create a dummy label image
#     _, dummy_label = image_label#np.random.randint(0, 8, (1024, 2048, 1), dtype=np.uint8)
#     dummy_label_path = 'tests/dummy_label.png'
#     tf.keras.preprocessing.image.save_img(dummy_label_path, dummy_label)

#     # Read the label using read_label function
#     label = read_label(dummy_label_path)

#     # Check that the label has the correct shape and dtype
#     assert label.shape == (1024, 2048, 1), "Label shape is incorrect."
#     assert label.dtype == tf.uint8, "Label dtype is not tf.uint8."

#     # Clean up: Remove the dummy label file
#     import os
#     os.remove(dummy_label_path)



# def test_normalize(image_label):
#     # Create dummy image and mask
#     input_image = image_label[0] 
#     #tf.random.uniform((1024, 2048, 3), minval=0, maxval=1, dtype=tf.float32)
#     input_mask = image_label[1] 
#     #tf.random.uniform((1024, 2048, 1), minval=0, maxval=8, dtype=tf.int32)

#     # Normalize the image
#     normalized_image, output_mask = normalize(input_image, input_mask)

#     # Check that normalized_image values are within [0, 1]
#     assert tf.reduce_min(normalized_image) >= 0.0, "Normalized image has values less than 0."
#     assert tf.reduce_max(normalized_image) <= 1.0, "Normalized image has values greater than 1."

#     # Check that the mask is unchanged
#     assert tf.reduce_all(tf.equal(input_mask, output_mask)), "Mask should remain unchanged after normalization."



# def test_retrieve_mask_mappings():
#     original_classes, class_mapping, new_labels = retrieve_mask_mappings()

#     # Check that the mappings have correct lengths
#     assert len(original_classes) == 30, "There should be 30 original classes."
#     assert len(class_mapping) == 30, "Class mapping should map 30 classes."
#     assert len(new_labels) == 8, "There should be 8 new labels."

#     # Check that all original classes are mapped
#     for cls in original_classes:
#         assert cls in class_mapping, f"Class '{cls}' is missing in class mapping."




# def test_map_labels_tf(image_label):
#     # Create a dummy label image with original class indices
#     original_classes, class_mapping, new_labels = retrieve_mask_mappings()
#     label_image = image_label[1]
#     # Map labels
#     mapped_label = map_labels_tf(label_image, original_classes, class_mapping, new_labels)

#     # Check that the mapped labels are within the new label range
#     unique_labels = tf.unique(tf.reshape(mapped_label, [-1]))[0]
#     assert tf.reduce_max(unique_labels) < 8, "Mapped labels should be less than 8."
#     assert tf.reduce_min(unique_labels) >= 0, "Mapped labels should be non-negative."



# def test_augment_image_and_label(image_label):
#     # Create dummy image and label
#     image = image_label[0] 
#     #tf.random.uniform((1024, 2048, 3), dtype=tf.float32)
#     label = image_label[1] 
#     #tf.random.uniform((1024, 2048, 1), minval=0, maxval=8, dtype=tf.uint8)

#     # Apply augmentation
#     augmented_image, augmented_label = augment_image_and_label(image, label, augment_prob=1.0)

#     # Check that the augmented image has the same shape and dtype
#     assert augmented_image.shape == image.shape, "Augmented image shape is incorrect."
#     assert augmented_image.dtype == tf.float32, "Augmented image dtype is not tf.float32."

#     # Check that the augmented label has the same shape and dtype
#     assert augmented_label.shape == label.shape, "Augmented label shape is incorrect."
#     assert augmented_label.dtype == tf.uint8, "Augmented label dtype is not tf.uint8."



# def test_resize_images(image_label):
#     # Create dummy image and label
#     image = image_label[0] 
#     #tf.random.uniform((1024, 2048, 3), dtype=tf.float32)
#     label = image_label[1]
#     #tf.random.uniform((1024, 2048, 1), minval=0, maxval=8, dtype=tf.uint8)

#     # Resize images and labels
#     resized_image, resized_label = resize_images(image, label)

#     # Check the new shapes
#     assert resized_image.shape == (3, 512, 1024), "Resized image shape is incorrect."
#     assert resized_label.shape == (1, 128, 256), "Resized label shape is incorrect."

#     # Check data types
#     assert resized_image.dtype == tf.float32, "Resized image dtype is not tf.float32."
#     assert resized_label.dtype == tf.uint8, "Resized label dtype is not tf.uint8."



# from unittest.mock import patch

# def test_load_dataset_unet(image_label):
#     # Mock get_datasets to return dummy paths
#     with patch('data.data_processing.get_datasets') as mock_get_datasets:
#         mock_get_datasets.return_value = {
#             "segmentation_images_train": ['tests/dummy_image.png'],
#             "segmentation_masks_train": ['tests/dummy_label.png'],
#             "segmentation_images_valid": ['tests/dummy_image.png'],
#             "segmentation_masks_valid": ['tests/dummy_label.png'],
#         }

#         # Ensure dummy files exist
#         dummy_image = image_label[0]#np.random.randint(0, 255, (1024, 2048, 3), dtype=np.uint8)
#         dummy_label = image_label[1]#np.random.randint(0, 8, (1024, 2048, 1), dtype=np.uint8)
#         tf.keras.preprocessing.image.save_img('tests/dummy_image.png', dummy_image)
#         tf.keras.preprocessing.image.save_img('tests/dummy_label.png', dummy_label)

#         # Call the function
#         dataset_train, dataset_valid = load_dataset_unet(batch_size=1)

#         # Fetch one batch to test
#         for image_batch, label_batch in dataset_train.take(1):
#             assert image_batch.shape == (1, 1024, 2048, 3), "Image batch shape is incorrect."
#             assert label_batch.shape == (1, 1024, 2048, 1), "Label batch shape is incorrect."

#         # Clean upz
#         import os
#         os.remove('tests/dummy_image.png')
#         os.remove('tests/dummy_label.png')
