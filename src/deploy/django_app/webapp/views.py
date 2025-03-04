import json
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# Dummy function to simulate model inference
def simulate_model_inference(image_name):
    # Replace this with your actual model inference logic.
    original_url = '/static/deploy/images/' + image_name
    prediction_url = '/static/deploy/images/' + image_name.replace('.jpg', '_mask.jpg')
    label = "Label for " + image_name
    return original_url, prediction_url, label

def index(request):
    # List of 20 test images. Adjust the names or sourcing as needed.
    image_list = [f'image{i}.jpg' for i in range(1, 21)]
    return render(request, 'deploy/index.html', {'images': image_list})

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_name = data.get('image_name')
            if not image_name:
                return HttpResponseBadRequest("Missing image_name parameter")
            original, prediction, label = simulate_model_inference(image_name)
            return JsonResponse({'original': original, 'prediction': prediction, 'label': label})
        except Exception as e:
            return HttpResponseBadRequest(str(e))
    return HttpResponseBadRequest("Only POST requests are allowed.")
