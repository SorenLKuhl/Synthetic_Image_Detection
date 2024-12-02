import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm
from models import resnet18,cnn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import random
import configparser


# Path to data
# dataset_path = "/dtu/blackhole/01/169729/stable-diffusion-face-dataset-main/512/"
# dataset_path = "/dtu/blackhole/01/169729/real_and_fake_faces/real_vs_fake/real-vs-fake/valid"
dataset_path = "/dtu/blackhole/01/169729/gan_validation/"
# dataset_path = "/dtu/blackhole/01/169729/mixed_dataset/train"


# Hyper-parameters
data_fraction = 1.0 # fraction of data to use.
num_epochs = 1
batch_size = 32

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Load trained model
model = cnn.get_model()
# model.load_state_dict(torch.load("output/saved/CNN_v1.pt", weights_only=True))
model.load_state_dict(torch.load("output/model.pt", weights_only=True))
# model = resnet18.get_model()
# model.load_state_dict(torch.load("output/saved/ResNet18.pt", weights_only=True))
model.eval()


# Data loading
# Transform
transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
# Full dataset
full_dataset = ImageFolder(dataset_path, transform=transform)
classes = full_dataset.classes
# Loader
data_loader = DataLoader(dataset=full_dataset, batch_size=32, shuffle=True)


# Loss
criterion = nn.CrossEntropyLoss()


# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))

# Test
for data, target in data_loader:
    # move tensors to GPU if CUDA is available
    data, target = data.to(device), target.to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    
    for i in range(len(target.data)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
        
# average test loss
test_loss = test_loss/len(data_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(2):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))