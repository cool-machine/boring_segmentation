import tensorflow as tf


@tf.function
def step(images, 
         masks, 
         model,
         total_loss,
         accuracy,
         loss_fn,
         optimizer,  
         training = False,              
         ):


    if training:
        with tf.GradientTape() as tape:
            # Get logits from the model
            logits = model(images, training=training).logits

            # Ensure logits have shape [batch_size, height, width, num_classes (in place of channels)]
            logits = tf.transpose(logits, perm=[0, 2, 3, 1])

            # Ensure masks shape is correct [batch_size, height, width, channels]
            masks = tf.transpose(masks, perm=[0, 2, 3, 1])
            
            # Remove the extra dimension:
            # from (B, 1, H, W) -> (B, H, W)
            if masks.shape.rank == 4 and masks.shape[-1] == 1:
                masks = tf.squeeze(masks, axis=-1)

            # Compute loss using logits directly
            loss = loss_fn(masks, logits)

            # Compute gradients and apply them
        gradients = tape.gradient(loss, model.trainable_variables)

        # scaled_gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    else: 
        # Get logits from the model
        logits = model(images, training=training).logits

        # Ensure logits have shape [batch_size, height, width, num_classes (in place of channels)]
        logits = tf.transpose(logits, perm=[0, 2, 3, 1])

        # Ensure masks shape is correct [batch_size, height, width, channels]
        masks = tf.transpose(masks, perm=[0, 2, 3, 1])
        
        # Remove the extra dimension:
        # from (B, 1, H, W) -> (B, H, W)
        if masks.shape.rank == 4 and masks.shape[-1] == 1:
            masks = tf.squeeze(masks, axis=-1)

        # Compute loss using logits directly
        loss = loss_fn(masks, logits)
 
    # Update metrics with predictions
    total_loss(loss)
    
    accuracy.update_state(masks, logits)

    return loss
