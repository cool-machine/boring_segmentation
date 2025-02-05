import os
import shutil
import mlflow

best_models = []

def maybe_save_best_model(model, epoch_val_loss, epoch):
    global best_models
    
    if len(best_models) < 3 or epoch_val_loss < max(m["val_loss"] for m in best_models):
        model_dir = "./outputs/segformer"
        os.makedirs(model_dir, exist_ok=True)

        # Create a *unique* subdir for each checkpoint
        checkpoint_dir = os.path.join(model_dir, f"epoch_{epoch}_val_{epoch_val_loss}")
        # Create the subfolder for this checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Now save_pretrained into this *folder*
        model.save_pretrained(checkpoint_dir)

        # Log the entire directory to MLflow (zips it as an artifact)
        mlflow.log_artifacts(checkpoint_dir, artifact_path=f"checkpoints/epoch_{epoch}")

        # Add the directory path to best_models
        best_models.append({
            "val_loss": epoch_val_loss,
            "epoch": epoch+1,
            "path": checkpoint_dir  # store the folder path
        })

        # Sort & remove the worst
        best_models.sort(key=lambda x: x["val_loss"])
        if len(best_models) > 3:
            to_remove = best_models.pop()
            if os.path.exists(to_remove["path"]):
                # Remove the entire checkpoint folder
                shutil.rmtree(to_remove["path"])