import os
from azureml.core import Dataset, Workspace
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K



def get_datasets():
    
    # Connect to your workspace
    ws = Workspace.from_config()

    # Dictionary to store datasets and their mount points
    datasets_paths = {}
    mount_points = {}
    
    # List of dataset names
    dataset_names = [
        ["segmentation_images_train", "train"],
        ["segmentation_masks_train", "_gtFine_labelIds.png"],
        ["segmentation_images_val", "val"],
        ["segmentation_masks_val", "_gtFine_labelIds.png"],
    ]
    # Mount each dataset
    for name in dataset_names:
        dataset = Dataset.get_by_name(ws, name[0])
        mount_point = dataset.mount()
        mount_point.start()

        paths = dataset.to_path()

        if name[1] == "_gtFine_labelIds.png":
            paths = [p for p in paths if name[1] in p]  
        datasets_paths[name[0]] = [os.path.join(mount_point.mount_point, path.lstrip("/")) for path in paths]
        mount_points[name[0]] = mount_point
        print(f"Dataset '{name}' mounted at '{mount_point.mount_point}'")
    return datasets_paths, mount_points




def check_paths(imgs, msks):
    for pair in zip(imgs, msks): 
        img_pth = pair[0].split("/")[-1].split("_")
        msk_pth = pair[0].split("/")[-1].split("_")
        assert img_pth == msk_pth, f"Paths of image {pair[0]} and mask {pair[1]} do not match"
    return "Paths are correct - check passed"    




def read_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape([1024, 2048, 3]) 
    return image




def read_label(file_path):
    label = tf.io.read_file(file_path)
    label = tf.image.decode_image(label, channels=1)
    label = tf.image.convert_image_dtype(label, tf.uint8)
    label.set_shape([1024, 2048])
    return label




def normalize(input_image, input_mask):
    '''
    Function normalize adjusts values of images to the values that 
    are expected by trained models (in this case Segformer). 
    It uses means and standard deviations 
    for each channel of image. 

    Remarque: labels do not need adjustment

    Arguments: 
        input_image: color image received by the function
        input_mask: mask that corresponds to the image
    
    '''
    # Hard-code values of mean and standard deviation 
    # required to adjust images for segformer
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    # Adjust pixel values for input images
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_image = (input_image - mean) / tf.maximum(std, K.epsilon())

    # Clip the pixel values of images to ensure they are within [0, 1]
    input_image = tf.clip_by_value(input_image, clip_value_min=0.0, clip_value_max=1.0)
    
    return input_image, input_mask




def retrieve_mask_mappings():
    
    # Original 30 classes
    original_classes = [
        'road', 'sidewalk', 'parking', 'rail track', 'person', 'rider', 'car', 'truck', 'bus', 'on rails',
        'motorcycle', 'bicycle', 'caravan', 'trailer', 'building', 'wall', 'fence', 'guard rail', 'bridge',
        'tunnel', 'pole', 'pole group', 'traffic sign', 'traffic light', 'vegetation', 'terrain', 'sky',
        'ground', 'dynamic', 'static'
    ]


    # Mapping to 8 major groups
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

    # New labels for the 8 major groups
    new_labels = {
        'flat': 0, 'human': 1, 
        'vehicle': 2, 'construction': 3, 
        'object': 4, 'nature': 5, 
        'sky': 6, 'void': 7
    }

    return original_classes, class_mapping, new_labels




def map_labels_tf(label_image, original_classes, class_mapping, new_labels):
    label_image = tf.squeeze(label_image)
    label_image_shape = tf.shape(label_image)
    mapped_label_image = tf.zeros_like(label_image, dtype=tf.uint8)
    for original_class, new_class in class_mapping.items():
        original_class_index = tf.cast(original_classes.index(original_class), tf.uint8)
        new_class_index = tf.cast(new_labels[new_class], tf.uint8)
        mask = tf.equal(label_image, original_class_index)
        fill_val = tf.fill(label_image_shape, tf.cast(new_class_index, tf.uint8))  
        mapped_label_image = tf.where(mask, fill_val, mapped_label_image)
    label = tf.expand_dims(mapped_label_image, axis=-1)  # Add back the last dimension
    label = tf.image.convert_image_dtype(label, tf.uint8)
    return label




def augment_image_and_label(image, label, augment_prob=0.3):
    
    def augment():
        # Generate a random seed
        seed = tf.random.uniform([2], maxval=64, dtype=tf.int32)
        # Apply augmentations with the seed
        if tf.random.uniform([], minval=0, maxval=1.0, dtype=tf.float32) > augment_prob:
            image_in = tf.image.stateless_random_flip_left_right(image, seed=seed)
            label_in = tf.image.stateless_random_flip_left_right(label, seed=seed)
        else:
            image_in = image
            label_in = label

        seed = tf.random.uniform([2], maxval=64, dtype=tf.int32)
        
        if tf.random.uniform([], minval=0, maxval=1.0, dtype=tf.float32) > augment_prob:
            image_in = tf.image.stateless_random_flip_up_down(image_in, seed=seed)
            label_in = tf.image.stateless_random_flip_up_down(label_in, seed=seed)

        if tf.random.uniform([], minval=0, maxval=1.0, dtype=tf.float32) > augment_prob:
            image_in = tf.image.stateless_random_brightness(image_in, max_delta=0.2, seed=seed)
            image_in = tf.image.stateless_random_contrast(image_in, lower=0.8, upper=1.2, seed=seed)

        return image_in, label_in

    def no_augment():
        return image, label

    # Use a stateless random number for consistency
    random_number = tf.random.uniform([], minval=0, maxval=1.0, dtype=tf.float32)
    
    image, label = tf.cond(random_number < augment_prob, augment, no_augment)

    image = tf.image.convert_image_dtype(image, tf.float32)

    # Clip the values to ensure they are within [0, 1]
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    # Make sure labels are of tf.uint8 format
    label = tf.image.convert_image_dtype(label, tf.uint8)
    
    return image, label




def load_dataset_unet(batch_size=2):
    
    # Get full paths for images and masks
    datasets, mount_points = get_datasets()

    train_image_paths = datasets["segmentation_images_train"]
    train_mask_paths = datasets["segmentation_masks_train"]
    
    val_image_paths = datasets["segmentation_images_val"]
    val_mask_paths = datasets["segmentation_masks_val"]
    
    # Create a dataset from the tuples of (image_path, mask_path)
    dataset_train = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mask_paths))
    dataset_valid = tf.data.Dataset.from_tensor_slices((val_image_paths, val_mask_paths))

    # Read image paths in tensorflow
    dataset_train = dataset_train.map(lambda image, mask: (read_image(image), read_label(mask)))
    dataset_valid = dataset_valid.map(lambda image, mask: (read_image(image), read_label(mask)))

    # Group 30 categories of labels into eight categories
    original_classes, class_mapping, new_labels = retrieve_mask_mappings()
    
    dataset_train = dataset_train.map(lambda image, mask: (image, map_labels_tf(mask, 
                                                                    original_classes, 
                                                                    class_mapping, 
                                                                    new_labels)),           
                                                           num_parallel_calls=tf.data.AUTOTUNE)

    dataset_valid = dataset_valid.map(lambda image, mask: (image, map_labels_tf(mask, 
                                                                    original_classes, 
                                                                    class_mapping, 
                                                                    new_labels)),           
                                                           num_parallel_calls=tf.data.AUTOTUNE)

    

    # Normalize examples
    dataset_train = dataset_train.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_valid = dataset_valid.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset_train = dataset_train.map(augment_image_and_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset_train = dataset_train.batch(batch_size, drop_remainder=True)
    dataset_valid= dataset_valid.batch(batch_size, drop_remainder=True)

    dataset_train = dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset_valid = dataset_valid.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset_train, dataset_valid, mount_points




def resize_images(image, label):

    image = tf.image.resize(image, size=[512, 1024], method=tf.image.ResizeMethod.BILINEAR)

    label = tf.image.resize(label, size=[128, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = tf.image.convert_image_dtype(label, tf.uint8)

    image = tf.transpose(image, perm=[2, 0, 1])
    output_mask = tf.transpose(label, perm=[2, 0, 1])

    output_image = tf.image.convert_image_dtype(image, tf.float32)
    output_image = tf.clip_by_value(output_image, clip_value_min=0.0, clip_value_max=1.0)
    
    return output_image, output_mask




def load_dataset_segf(dataset, batch_size=2):

    # Parse the serialized examples
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(augment_image_and_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset