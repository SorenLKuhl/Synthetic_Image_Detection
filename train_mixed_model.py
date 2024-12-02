# Imports
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm
from models import resnet18,cnn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os
import random
import configparser


# Path to blackhole dir
config = configparser.ConfigParser()
config.read('config.ini')
blackhole_path = config.get('BLACKHOLE', 'path')

# Name of dataset
dataset_name = "mixed_dataset/train"
# Dataset path
dataset_path = blackhole_path + dataset_name


# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Hyper-parameters
data_fraction = 1.0 # fraction of data to use.
learning_rate = 0.001
num_epochs = 10
batch_size = 32


# Data loading. 
# Transform
transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
# Full dataset
full_dataset = ImageFolder(dataset_path, transform=transform)
# Split
train_dataset, test_dataset = random_split(full_dataset, [0.9, 0.1])
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
    # for images, labels in train_loader:
    for images, labels in tqdm.tqdm(train_loader):  # Progress bar in console.
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
best_accuracy = 0
best_model = copy.deepcopy(model)
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    # save model if it is the best performing so far
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model = copy.deepcopy(model)
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, "
          f"Test Loss: {test_loss:.4f}, "
          f"Test Accuracy: {test_accuracy:.2f}%")
    
print("Highest accuracy achieved: " + str(best_accuracy))
torch.save(best_model.state_dict(), "output/model.pt")