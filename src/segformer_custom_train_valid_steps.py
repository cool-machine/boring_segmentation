import tensorflow as tf


@tf.function
def train_step(images, masks):
    with tf.GradientTape() as tape:
        # Get logits from the model
        logits = model(images, training=True).logits

        # Ensure logits have shape [batch_size, height, width, num_classes]
        logits = tf.transpose(logits, perm=[0, 2, 3, 1])

        # Ensure masks shape is correct
        masks = tf.squeeze(masks) #, axis=-1)

        # Compute loss using logits directly
        loss = loss_fn(masks, logits)

    # Compute gradients and apply them
    # scaled_loss = optimizer.get_scaled_loss(loss)
    gradients = tape.gradient(loss, model.trainable_variables)

    # scaled_gradients = optimizer.get_unscaled_gradients(gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Update metrics with predictions (using argmax to get predicted classes)
    train_loss(loss)
    train_accuracy.update_state(masks, logits)
    return loss

@tf.function
def val_step(images, masks):
    # Get logits from the model
    # with tf.device('/CPU:0'):  # Ensure validation is done on CPU
    logits = model(images, training=False).logits

    # Ensure logits have shape [batch_size, height, width, num_classes]
    logits = tf.transpose(logits, perm=[0, 2, 3, 1])

    # Remove the last dimension of masks if it exists
    masks = tf.squeeze(masks) #, axis=-1)

    # Compute loss
    loss = loss_fn(masks, logits)

    # Update metrics with predictions (using argmax to get predicted classes)
    val_loss(loss)
    val_accuracy.update_state(masks, logits)
    return loss