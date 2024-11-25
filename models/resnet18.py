# Model for comparison

# Imports
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights



# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


model = resnet18(ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes -> real and fake.
model = model.to(device)

def get_model():
    return model