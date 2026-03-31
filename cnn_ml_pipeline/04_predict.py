import torch
from torchvision import models, transforms
from PIL import Image
import pickle
import argparse
import os

def predict_single_image(image_path, ml_model_name="logistic_regression"):
    device = torch.device("cpu")
    
    # 1. Load the Pretrained Feature Extractor (MobileNetV2)
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = torch.nn.Identity() # Remove top classification layer
    model.eval()
    model = model.to(device)

    # 2. Setup Image Transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Process the Image
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return

    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension

    # 4. Extract Feature Vector
    with torch.no_grad():
        feature_vector = model(input_tensor).cpu().numpy()

    # 5. Load the ML Model
    model_path = f'models/{ml_model_name}_model.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Did you train it?")
        return
        
    with open(model_path, 'rb') as f:
        ml_model = pickle.load(f)

    # 6. Predict the Class
    prediction = ml_model.predict(feature_vector)
    
    # Normally we load the train dataset classes to map ID to string label. 
    # For dummy datasets, 0 = clean_water, 1 = polluted_water.
    # In a real app, save class names during training and load here.
    class_names = ['clean_water', 'polluted_water'] 
    predicted_class = class_names[prediction[0]]

    print(f"Prediction for '{image_path}':")
    print(f"Using ML Model: {ml_model_name}")
    print(f"Result: {predicted_class} (Class ID: {prediction[0]})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict class of an image using extracted Features + ML model")
    parser.add_argument("image_path", type=str, help="Path to the image to classify")
    parser.add_argument("--model", type=str, default="logistic_regression", choices=["svm", "random_forest", "logistic_regression"], help="ML model to use")
    
    args = parser.parse_args()
    predict_single_image(args.image_path, args.model)
