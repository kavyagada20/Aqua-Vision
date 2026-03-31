import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os
import joblib

def main():
    print("========== TRAINING CNN + LOGISTIC REGRESSION ==========")
    if not os.path.exists('dataset'):
        print("Error: Could not find 'dataset' folder. Please create it and add 'class' subfolders with images inside.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder('dataset', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    print(f"Loaded {len(dataset)} images from classes: {dataset.classes}")

    model = models.mobilenet_v2(pretrained=True)
    model.classifier = torch.nn.Identity()
    model.eval()

    features_list, labels_list = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            extracted = model(inputs)
            features_list.append(extracted.cpu().numpy())
            labels_list.append(targets.numpy())

    features_matrix = np.vstack(features_list)
    labels_array = np.concatenate(labels_list)
    
    print("\nTraining Logistic Regression Model...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(features_matrix, labels_array)
    
    predictions = lr_model.predict(features_matrix)
    acc = accuracy_score(labels_array, predictions)
    print(f"Logistic Regression Accuracy on Dataset: {(acc*100):.2f}%")
    
    print("\n========== EVALUATION METRICS ==========")
    print("Classification Report (Precision, Recall, F1-Score):")
    print(classification_report(labels_array, predictions, target_names=dataset.classes, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(labels_array, predictions))

    joblib.dump(lr_model, 'cnn_lr_model.pkl')
    print("\nSaved Logistic Regression model to 'cnn_lr_model.pkl'")

if __name__ == "__main__":
    main()
