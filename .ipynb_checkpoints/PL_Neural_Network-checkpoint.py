import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

class PersistenceLandscapeNN(nn.Module):
    def __init__(self):
        super(PersistenceLandscapeNN, self).__init__()

        # Linear Layers (in order)
        self.layer1 = nn.Linear(500, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, 16)
        self.layer6 = nn.Linear(16, 8)
        self.layer7 = nn.Linear(8, 4)

        # Activation Functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.sigmoid(x)
        x = self.layer2(x)
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
        x = self.softmax(x)
        return x
        