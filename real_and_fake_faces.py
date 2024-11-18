# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models import resnet18,cnn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import random



# Path to blackhole dir
blackhole_path = "/dtu/blackhole/01/169729/"
# Name of dataset
dataset_name = "real_and_fake_faces/real_vs_fake/real-vs-fake"
# Dataset path
dataset_path = blackhole_path + dataset_name


# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Hyper-parameters
data_fraction = 0.1 # fraction of data to use.
learning_rate = 0.01
num_epochs = 100
batch_size = 32


# Data loading. Inspiration from https://www.kaggle.com/code/nicoladisabato/fake-face-detection-with-keras-accuracy-0-987#Data-Loading and https://www.kaggle.com/code/sukhdev1234/deepfakes-recognition-btech-project-49d8ce#2.1.-Network-Architectures
# Transform
transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
# Directories:
train_dir = os.path.join(dataset_path, 'train')
valid_dir = os.path.join(dataset_path, 'valid')
test_dir = os.path.join(dataset_path, 'test')
# Full datasets
full_train_dataset = ImageFolder(train_dir, transform=transform)
full_valid_dataset = ImageFolder(valid_dir, transform=transform)
full_test_dataset = ImageFolder(test_dir, transform=transform)
# Determine the number of objects to be selected
num_train_data = int(len(full_train_dataset) * data_fraction)
num_test_data = int(len(full_test_dataset) * data_fraction)
# Randomly select objects for the datasets
train_indices = random.sample(range(len(full_train_dataset)), num_train_data)
test_indices = random.sample(range(len(full_test_dataset)), num_test_data)
# Create datasets with randomly selected objects
train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
test_dataset = torch.utils.data.Subset(full_test_dataset, test_indices)
# Loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Model
model = cnn.get_model()


# Loss
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    return running_loss / len(train_loader.dataset)

# Evaluation Function
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return running_loss / len(test_loader.dataset), accuracy

# Training the Model
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, "
          f"Test Loss: {test_loss:.4f}, "
          f"Test Accuracy: {test_accuracy:.2f}%")