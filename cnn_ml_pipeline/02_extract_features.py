import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import os

def extract_features():
    device = torch.device("cpu")
    print("Extracting features using MobileNetV2...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder('dataset', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    model = models.mobilenet_v2(pretrained=True)
    # Remove the classifier head, only keep feature extraction
    model.classifier = torch.nn.Identity()
    model.eval()
    model = model.to(device)

    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            # Forward pass - produces tensor of shape (batch, 1280)
            outputs = model(inputs)
            
            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())

    features = np.vstack(features)
    labels = np.concatenate(labels)

    os.makedirs('features', exist_ok=True)
    np.save('features/features.npy', features)
    np.save('features/labels.npy', labels)

    print(f"Extracted features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print("Features saved to 'features/features.npy' and 'features/labels.npy'.")

if __name__ == "__main__":
    extract_features()
