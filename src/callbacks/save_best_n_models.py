import os
import sys
import mlflow

# Keep a global (or outer-scope) list of best models
best_models = []  # each item is dict: {"val_loss": float, "epoch": int, "path": str}

def maybe_save_best_model(model, epoch_val_loss, epoch):
    """
    Save and log the current model if it ranks in the top 3 (lowest val_loss).
    Remove the worst checkpoint if we exceed 3.
    """
    global best_models

    # 1) If we have fewer than 3, always add
    # 2) Otherwise, only add if this val_loss is better than the worst (i.e., largest val_loss) in best_models
    if len(best_models) < 3 or epoch_val_loss < max(m["val_loss"] for m in best_models):
        # Save this model checkpoint locally
        model_dir = "./outputs/segformer"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"epoch-{epoch}-val_loss-{epoch_val_loss}.h5")

        model.save_pretrained(model_path)
        # model.save_pretrained("./segformer_saved")
        # Log it to MLflow as an artifact (the 'artifact_path' groups them under "checkpoints/")
        mlflow.log_artifact(model_path, artifact_path="checkpoints")

        # Add it to our best_models list
        best_models.append({
            "val_loss": epoch_val_loss,
            "epoch": epoch+1,
            "path": model_dir
        })

        # Sort best_models by ascending val_loss
        best_models.sort(key=lambda x: x["val_loss"])

        # If we now have > 3, remove the worst (the last in the sorted list)
        if len(best_models) > 3:
            to_remove = best_models.pop()  # remove the worst (highest val_loss)
            # Optionally remove the local file to save disk space
            if os.path.exists(to_remove["path"]):
                os.remove(to_remove["path"])
