#!/usr/bin/env python
import os
import argparse
import tensorflow as tf
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Patch call_context if needed.
if not hasattr(tf.keras.backend, "call_context"):
    tf.keras.backend.call_context = lambda: type("DummyContext", (), {"in_call": False})()

from src.data.processor import load_dataset_segf
from src.params.architectures.segformer import segformer
from src.params.loss_funcs import sparse_categorical_crossentropy_loss

def evaluate_model(model, test_dataset):
    """
    Evaluate the model on the test dataset and print the average loss.
    
    This function assumes:
      - Model outputs logits in channels-first format: [batch, num_classes, H, W]
      - We transpose logits to channels-last: [batch, H, W, num_classes]
      - Ground-truth masks are initially [batch, 1, H, W] and need to be reshaped to [batch, H, W]
    """
    loss_fn = sparse_categorical_crossentropy_loss()
    total_loss = 0.0
    count = 0
    for images, masks in test_dataset:
        # Run model prediction in inference mode.
        outputs = model(images, training=False)
        logits = outputs.logits
        # Transpose logits from [batch, num_classes, H, W] to [batch, H, W, num_classes]
        logits = tf.transpose(logits, perm=[0, 2, 3, 1])
        # Reshape masks from [batch, 1, H, W] to [batch, H, W]
        batch = tf.shape(masks)[0]
        height = tf.shape(masks)[2]
        width = tf.shape(masks)[3]
        masks = tf.reshape(masks, (batch, height, width))
        # Compute loss.
        loss = loss_fn(masks, logits)
        total_loss += loss.numpy()
        count += 1
    avg_loss = total_loss / count if count > 0 else float('nan')
    print("Average test loss:", avg_loss)
    return avg_loss

def main():
    parser = argparse.ArgumentParser(
        description="Validate the final model on the test dataset."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the final model checkpoint directory."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for testing."
    )
    args = parser.parse_args()

    # Load the test dataset. load_dataset_segf returns (train_ds, valid_ds, test_ds)
    _, _, test_dataset = load_dataset_segf(
        train_batch_size=args.batch_size,
        valid_batch_size=args.batch_size
    )

    # Load the model from the checkpoint. Set initial=False so that it loads from the given path.
    model = segformer(initial=False, path=args.model_path)

    # Evaluate the model on the test dataset.
    evaluate_model(model, test_dataset)

if __name__ == "__main__":
    main()
