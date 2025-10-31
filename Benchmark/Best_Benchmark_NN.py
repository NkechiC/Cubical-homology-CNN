import torch
import torch.nn as nn

class BenchmarkNN(nn.Module):
    def __init__(self):
        super(BenchmarkNN, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=7,
            stride=1,
            padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=4,
            kernel_size=10,
            stride=10,
            padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channels=4,
            out_channels=1,
            kernel_size=2,
            stride=1,
            padding=0
        )
        
        # Linear Layers
        self.layer1 = nn.Linear(16, 8)
        self.layer2 = nn.Linear(8, 5)
        self.layer3 = nn.Linear(5, 4)

        # Activation Layers
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # 2D Activation Layers
        self.avg_pool = nn.AvgPool2d(
            kernel_size=5,
            stride=5
        )

        # Dropout Layer (For Training)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.softmax(x)

        return x