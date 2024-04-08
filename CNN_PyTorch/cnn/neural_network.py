#from keras.models import Sequential
#from keras.layers import Conv2D
#from keras.layers import MaxPooling2D
#from keras.layers import Activation
#from keras.layers import Flatten
#from keras.layers import Dense
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, width, height, depth, total_classes):
        super(CNN, self).__init__()
        
        # First CONV => RELU => POOL Layer
        self.conv1 = nn.Conv2d(depth, 20, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1)
        
        # Second CONV => RELU => POOL Layer
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=1, stride=1)
        
        # Third CONV => RELU => POOL Layer
        self.conv3 = nn.Conv2d(50, 100, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=1, stride=1)
        
        # Fully connected layer
        self.fc = nn.Linear(width * height * 100, total_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x

                  
    
# --------------------------------- EOC ------------------------------------
