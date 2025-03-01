import torch
import json
from azureml.core.model import Model

def init():
    global model, device
    model_path = Model.get_model_path("best_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

def run(data):
    try:
        inputs = json.loads(data)["data"]
        tensor_input = torch.tensor(inputs, dtype=torch.float32).to(device)
        predictions = model(tensor_input).cpu().detach().numpy().tolist()
        return {"predictions": predictions}
    except Exception as e:
        return {"error": str(e)}
