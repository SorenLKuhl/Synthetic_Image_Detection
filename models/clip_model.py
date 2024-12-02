import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
from PIL import Image
import configparser

# Path to blackhole dir
config = configparser.ConfigParser()
config.read('config.ini')
blackhole_path = config.get('BLACKHOLE', 'path')


# Step 1: Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Step 2: Prepare dataset
# Dataset path should have two subfolders: `class1` and `class2`
data_dir = "dataset"  # Replace with the path to your dataset
transform = preprocess  # Use CLIP's preprocessing

# Split dataset into train and validation sets
train_dataset = ImageFolder(root=f"{blackhole_path}/real_vs_fake/real-vs-fake/train", transform=transform)
test_dataset = ImageFolder(root=f"{blackhole_path}/real_vs_fake/real-vs-fake/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 3: Define the fine-tuning model
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(CLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(512, num_classes)  # CLIP ViT-B/32 outputs 512-dim features

    def forward(self, images):
        with torch.no_grad():  # Keep CLIP's image encoder frozen
            features = self.clip_model.encode_image(images)
        features = features.float()  # Ensure the tensor is in float32
        features = features / features.norm(dim=-1, keepdim=True)  # Normalize features
        return self.classifier(features)

num_classes = 2  # Binary classification
model = CLIPClassifier(clip_model, num_classes).to(device)

# Step 4: Set up training components
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)  # Only optimize the classifier

# Step 5: Training loop
epochs = 10
best_test_accuracy = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    # Training
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total
    print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    # Validation
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total
    print(f"Epoch {epoch + 1}: test Loss: {test_loss:.4f}, test Accuracy: {test_accuracy:.4f}")

    # Save the best model
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save(model.state_dict(), "clip_binary_classifier.pth")
        print(f"Model saved with test Accuracy: {test_accuracy:.4f}")

# Step 6: Evaluate on the validation set
print(f"Best Test Accuracy: {best_test_accuracy:.4f}")
