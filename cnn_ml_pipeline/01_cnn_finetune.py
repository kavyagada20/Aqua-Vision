import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

def train_cnn():
    device = torch.device("cpu")
    print("Training purely on CNN (MobileNetV2) using CPU...")

    # Data transformation for MobileNetV2
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder('dataset', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    num_classes = len(dataset.classes)

    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(pretrained=True)
    
    # Freeze the feature extraction layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classification head
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    print(f"Classes found: {dataset.classes}")
    epochs = 2
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_acc = correct.double() / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(dataloader):.4f} | Acc: {epoch_acc:.4f}")

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/cnn_only_model.pth')
    print("Finished CNN Training. Model saved to 'models/cnn_only_model.pth'.")

if __name__ == "__main__":
    train_cnn()
