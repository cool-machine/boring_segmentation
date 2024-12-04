import numpy as np
import matplotlib.pyplot as plt


def plot_image_unet(image, segmentation, image_title='Original Image'):

    # Convert the image to uint8 if it's not already
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Ensure image is in the range [0, 1] and convert to uint8
    image = np.array(image)
    image = np.clip(image, 0, 1)  # Ensure values are in [0, 1]
    image = (image * 255).astype(np.uint8)

    # Convert segmentation to uint8
    segmentation = (segmentation.numpy()).astype(np.uint8)
    segmentation = np.squeeze(segmentation)
    segmentation = Image.fromarray(segmentation)

    # Convert to PIL Image
    image = Image.fromarray(image)

    # Plotting the image and its segmentation
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title(image_title)
    axes[0].axis('off')

    # Segmentation mask
    axes[1].imshow(image)
    axes[1].imshow(segmentation, cmap='jet', alpha=0.5)  # Overlaying segmentation with some transparency
    axes[1].set_title('Segmentation Applied')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    


def plot_image_segf(image, segmentation, image_title='Original Image'):
    # Convert the image to uint8 if it's not already
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.transpose(image, perm=[1, 2, 0])
    image = tf.image.resize(image, size=[128, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Ensure image is in the range [0, 1] and convert to uint8
    image = np.clip(image, 0, 1)  # Ensure values are in [0, 1]
    image = np.array(image)
    image = (image * 255).astype(np.uint8)

    # Convert image to RGB if it's in BGR
    if image.shape[-1] == 3:
        image = image[..., ::-1]  # Convert BGR to RGB if needed

    # Convert segmentation to uint8
    segmentation = tf.transpose(segmentation, perm=[1, 2, 0])
    segmentation = np.array(segmentation)
    # segmentation = (segmentation * 255).astype(np.uint8)

    # Convert to PIL Image
    image = Image.fromarray(image)

    # Plotting the image and its segmentation
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title(image_title)
    axes[0].axis('off')

    # Segmentation mask
    axes[1].imshow(image)
    axes[1].imshow(segmentation, cmap='jet', alpha=0.5)  # Overlaying segmentation with some transparency
    axes[1].set_title('Segmentation Applied')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def plot_predictions(image, mask, predictions, epoch):

    # Convert predictions to class labels
    predictions = tf.transpose(predictions, perm=[1, 2, 0])
    prediction = tf.argmax(predictions, axis=-1)

    image = tf.transpose(image, perm=[1, 2, 0])
    mask = tf.transpose(mask, perm=[1, 2, 0])

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[128, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    axes[0].imshow(image)
    axes[0].set_title(f'Epoch {epoch} - Input Image')
    axes[0].axis('off')


    # Segmentation mask
    axes[1].imshow(image)
    axes[1].imshow(mask, cmap='jet', alpha=0.5)  # Overlaying segmentation with some transparency
    axes[1].set_title('True Mask')
    axes[1].axis('off')

    # Segmentation mask
    axes[2].imshow(image)
    axes[2].imshow(prediction, cmap='jet', alpha=0.5)  # Overlaying segmentation with some transparency
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')


    plt.tight_layout()
    plt.savefig(f'predictions_epoch_{epoch}.png')
    plt.show()


