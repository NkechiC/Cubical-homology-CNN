import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

class PersistenceLandscapeNN(nn.Module):
    def __init__(self):
        super(PersistenceLandscapeNN, self).__init__()

        # Linear Layers (in order)
        self.layer1 = nn.Linear(18, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 32)
        self.layer4 = nn.Linear(32, 32)
        self.layer5 = nn.Linear(32, 32)
        self.layer6 = nn.Linear(32, 32)
        self.layer7 = nn.Linear(32, 32)
        self.layer8 = nn.Linear(32, 32)
        self.layer9 = nn.Linear(32, 32)
        self.layer10 = nn.Linear(32, 4)

        # Batch Norm Layers
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)

        # Activation Functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.relu(x)
        x = self.layer5(x)
        x = self.relu(x)
        x = self.layer6(x)
        x = self.relu(x)
        x = self.layer7(x)
        x = self.relu(x)
        x = self.layer8(x)
        x = self.relu(x)
        x = self.layer9(x)
        x = self.relu(x)
        x = self.layer10(x)

        if not self.training:
            x = self.softmax(x)
        
        return x