# Our model

# Imports
import torch
import torch.nn as nn

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        
        self.dropout_percentage = 0.2
        
        # Block 1
        # input 3x256x256
        # output 64x128x128
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        
        # Block 2
        # input 64x128x128
        # output 64x128x128
        self.conv2_1 = nn.Conv2d(64, 64, (3,3), 1, 1)
        self.batchnorm2_1 = nn.BatchNorm2d(64)
        self.act2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(64, 64, (3,3), 1, 1)
        self.batchnorm2_2 = nn.BatchNorm2d(64)
        self.act2_2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.dropout_percentage)
        

        # Block 3
        # input 64x128x128
        # output 128x64x64
        self.conv3_1 = nn.Conv2d(64, 128, (3,3), (2,2), (1,1))
        self.batchnorm3_1 = nn.BatchNorm2d(128)
        self.act3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(128, 128, (3,3), (1,1), (1,1))
        self.batchnorm3_2 = nn.BatchNorm2d(128)
        self.act3_2 = nn.ReLU()
        self.dropout3 = nn.Dropout(self.dropout_percentage)
        
        
        # Block 4
        # input 128x64x64
        # output 128x32x32
        self.conv4_1 = nn.Conv2d(128, 256, (3,3), (2,2), (1,1))
        self.batchnorm4_1 = nn.BatchNorm2d(256)
        self.act4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(256, 128, (3,3), (1,1), (1,1))
        self.batchnorm4_2 = nn.BatchNorm2d(128)
        self.act4_2 = nn.ReLU()
        self.dropout4 = nn.Dropout(self.dropout_percentage)
        
        
        # Block 5
        # input 128x32x32
        # output 64x16x16
        self.conv5_1 = nn.Conv2d(128, 64, (3,3), (2,2), (1,1))
        self.batchnorm5_1 = nn.BatchNorm2d(64)
        self.act5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(64, 64, (3,3), (1,1), (1,1))
        self.batchnorm5_2 = nn.BatchNorm2d(64)
        self.act5_2 = nn.ReLU()
        self.dropout5 = nn.Dropout(self.dropout_percentage)
        
        # Final Block
        # Flatten for ffnn: input 64x16x16, output 64x16x16
        self.flat = nn.Flatten()
        # Feed Foward Neural-Network
        # input 64x16x16, output num_classes
        self.fc = nn.Linear(64*16*16, num_classes)

        
    def forward(self, x):
        # Block #1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        # Block #2
        x = self.conv2_1(x)
        x = self.batchnorm2_1(x)
        x = self.act2_1(x)
        x = self.conv2_2(x)
        x = self.batchnorm2_2(x)
        x = self.act2_2(x)
        x = self.dropout2(x)
        
        # Block #3
        x = self.conv3_1(x)
        x = self.batchnorm3_1(x)
        x = self.act3_1(x)
        x = self.conv3_2(x)
        x = self.batchnorm3_2(x)
        x = self.act3_2(x)
        x = self.dropout3(x)
        
        # Block #4
        x = self.conv4_1(x)
        x = self.batchnorm4_1(x)
        x = self.act4_1(x)
        x = self.conv4_2(x)
        x = self.batchnorm4_2(x)
        x = self.act4_2(x)
        x = self.dropout4(x)
        
        # Block #5
        x = self.conv5_1(x)
        x = self.batchnorm5_1(x)
        x = self.act5_1(x)
        x = self.conv5_2(x)
        x = self.batchnorm5_2(x)
        x = self.act5_2(x)
        x = self.dropout5(x)
        
        # Final block
        x = self.flat(x)
        x = self.fc(x)     
        
        return x

def get_model():
	model = Model(2)
	model.to(device)
	return model