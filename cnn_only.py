import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import os

def main():
    print("========== TRAINING PURE CNN (MobileNetV2) ==========")
    if not os.path.exists('dataset'):
        print("Error: Could not find 'dataset' folder. Please create it and add 'class' subfolders with images inside.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder('dataset', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    num_classes = len(dataset.classes)
    print(f"Loaded {len(dataset)} images belonging to {num_classes} classes: {dataset.classes}")

    # Load pretrained CNN
    model = models.mobilenet_v2(pretrained=True)
    # Freeze the feature layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace classification layer
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
        acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(dataloader):.4f} | Accuracy: {(acc*100):.2f}%")
    
    # Final Evaluation Loop for Metrics
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n========== EVALUATION METRICS ==========")
    print("Classification Report (Precision, Recall, F1-Score):")
    print(classification_report(all_labels, all_preds, target_names=dataset.classes, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    torch.save(model.state_dict(), 'cnn_only_model.pth')
    print("\nTraining finished! Saved model as 'cnn_only_model.pth'")

if __name__ == "__main__":
    main()
