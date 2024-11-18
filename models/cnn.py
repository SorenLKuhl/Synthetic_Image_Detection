# Our model

# Imports
import torch
import torch.nn as nn

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.num_classes = num_classes
        
        # Convulutional layer #1: input 3x32x32, output 32x32x32
        self.conv1 = nn.Conv2d(3, 32, (3,3), 1, 1)
        self.act1 = nn.ReLU()
        
        #2x2 pooling: input 32x32x32, output 32x16x16
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Convulutional layer #2: input 32x16x16, output 32x16x16
        self.conv2 = nn.Conv2d(32, 32, (3,3), 1, 1)
        self.act2 = nn.ReLU()
        
        #2x2 pooling: input 32x32x32, output 32x8x8
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Convulutional layer #3: input 32x8x8, output 32x8x8
        self.conv3 = nn.Conv2d(32, 32, (3,3), 1, 1)
        self.act3 = nn.ReLU()
        
        # Flatten for ffnn: input 32x8x8, output 32x8x8
        self.flat = nn.Flatten()
        
        # Feed Foward Neural-Network
        # input 32x8x8, output 512
        self.fc3 = nn.Linear(32*8*8, 10)
        self.act3 = nn.ReLU()
        
        # input 512, output 10
        self.fc4 = nn.Linear(512, 10)
        self.act4 = nn.ReLU()
        
    def forward(self, x):
        # Convulutional layer #1
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        # Convulutional layer #2
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        # Convulutional layer #3
        x = self.conv3(x)
        x = self.act3(x)
		# Flatten
        x = self.flat(x)
        # FFNN #1
        x = self.fc3(x)
        # x = self.act3(x)
        # FFNN #2
        # x = self.fc4(x)
        # x = self.act4(x)
        return x

def get_model():
	model = Model(2)
	model.to(device)
	return model