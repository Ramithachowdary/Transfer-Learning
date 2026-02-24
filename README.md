# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for binary image classification (Defect vs Non-Defect) using VGG-19 architecture.

## Problem Statement and Dataset
In semiconductor or industrial manufacturing, identifying defective chips is critical for quality control. This experiment implements Transfer Learning using the pre-trained VGG-19 model to classify chip images into two categories:
- **Defect** – Chips with visible surface defects
- **Non-Defect** – Chips with no defects

The dataset (`chip_data`) consists of labelled images organized into `train` and `test` folders, each containing subfolders named `defect` and `non_defect`. The VGG-19 model pre-trained on ImageNet is fine-tuned by replacing its final classification layer to suit this binary classification task.

## DESIGN STEPS

### STEP 1: Data Loading and Preprocessing
Load the chip image dataset using `torchvision.datasets.ImageFolder`. Apply transformations including resizing all images to 224×224 (required for VGG-19 input), converting to tensors, and normalizing using ImageNet mean and standard deviation values.

### STEP 2: Load Pretrained VGG-19 and Modify for Transfer Learning
Load the VGG-19 model with pretrained ImageNet weights. Freeze all the convolutional feature extractor layers to retain learned features. Replace the final fully connected layer of the classifier with a new linear layer having output size equal to the number of classes (2 — defect and non-defect).

### STEP 3: Train, Evaluate and Predict
Define CrossEntropyLoss as the loss function and Adam optimizer acting only on the unfrozen classifier parameters. Train the model for 10 epochs while tracking training and validation loss. Evaluate the model on the test set by computing accuracy, plotting the confusion matrix, generating a classification report, and predicting on individual sample images.

## PROGRAM
```python
# ============================================================
# EX04: Implementation of Transfer Learning using VGG-19
# ============================================================

!pip install torchsummary -q

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torchsummary import summary

# ============================================================
# Step 1: Load and Preprocess Data
# ============================================================

!unzip -qq ./chip_data.zip -d data

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset_path = "./data/dataset/"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset  = datasets.ImageFolder(root=f"{dataset_path}/test",  transform=transform)

print(f"Classes: {train_dataset.classes}")
print(f"Total training samples : {len(train_dataset)}")
print(f"Total testing  samples : {len(test_dataset)}")

first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")

def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 4))
    for i in range(num_images):
        image, label = dataset[i]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        image = image * std + mean
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.suptitle("Sample Training Images")
    plt.tight_layout()
    plt.show()

show_sample_images(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# ============================================================
# Step 2: Load Pretrained VGG-19 and Modify for Transfer Learning
# ============================================================

model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

summary(model, input_size=(3, 224, 224))

# Modify the final fully connected layer to match the dataset classes
num_classes = len(train_dataset.classes)
model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

model = model.to(device)
summary(model, input_size=(3, 224, 224))

# Freeze all layers except the final layer
for param in model.features.parameters():
    param.requires_grad = False

# Include the Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# ============================================================
# Step 3: Train the Model
# ============================================================

def train_model(model, train_loader, test_loader, num_epochs=10):
    train_losses = []
    val_losses   = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(test_loader))

        print(f"Epoch [{epoch+1}/{num_epochs}]  "
              f"Train Loss: {train_losses[-1]:.4f}  "
              f"Val Loss: {val_losses[-1]:.4f}")

    print("Name:            ")
    print("Register Number: ")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss',      marker='o')
    plt.plot(range(1, num_epochs+1), val_losses,   label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return train_losses, val_losses

# Train the model
train_losses, val_losses = train_model(model, train_loader, test_loader, num_epochs=10)

# ============================================================
# Step 4: Test the Model
# ============================================================

def test_model(model, test_loader):
    model.eval()
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f"\nTest Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("Name:            ")
    print("Register Number: ")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    print("Name:            ")
    print("Register Number: ")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=train_dataset.classes))

test_model(model, test_loader)

# ============================================================
# Step 5: Predict on a Single Image
# ============================================================

def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]

    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted = predicted.item()

    class_names = dataset.classes

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img_display = image * std + mean
    img_display = img_display.permute(1, 2, 0).numpy()
    img_display = np.clip(img_display, 0, 1)

    plt.figure(figsize=(4, 4))
    plt.imshow(img_display)
    plt.title(f"Actual: {class_names[label]}\nPredicted: {class_names[predicted]}")
    plt.axis("off")
    plt.show()

    print(f"Actual: {class_names[label]}, Predicted: {class_names[predicted]}")

predict_image(model, image_index=55, dataset=test_dataset)
predict_image(model, image_index=25, dataset=test_dataset)
```

## OUTPUT
### Sample Input Images 
<img width="739" height="186" alt="image" src="https://github.com/user-attachments/assets/4f158d78-c6d1-4f19-8a96-f76e67626078" />

### Training Loss, Validation Loss Vs Iteration Plot
<img width="761" height="613" alt="image" src="https://github.com/user-attachments/assets/888d105b-a18b-42d0-93a2-2c1e50a311dc" />

### Confusion Matrix
<img width="756" height="664" alt="image" src="https://github.com/user-attachments/assets/7c21b539-beda-43bf-8bcd-44617f5e1bc1" />

### Classification Report
<img width="512" height="262" alt="image" src="https://github.com/user-attachments/assets/446f0499-c246-4c66-92d3-0df7b2ec2b97" />

### New Sample Prediction
<img width="339" height="786" alt="image" src="https://github.com/user-attachments/assets/8753ecd8-7ea7-48df-9f16-2f1ba40be02d" />

## RESULT
Transfer Learning using the pre-trained VGG-19 architecture was successfully implemented for binary classification of chip images into defect and non-defect categories. The model was fine-tuned by freezing the feature extraction layers and retraining only the final classifier layer. The trained model achieved good classification accuracy on the test dataset, as validated by the confusion matrix and classification report.
