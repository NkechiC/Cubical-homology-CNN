import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold
from PL_Neural_Network import PersistenceLandscapeNN

def Standardize(data, index):
    arr = data[index]
    mean = np.mean(arr)
    std = np.std(arr)
    mean_array = np.full(len(arr), mean)
    result = (arr - mean_array) * (1 / std)
    data[index] = result

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

training_data = pd.read_csv("brain_mri_dataset/training.csv")
training_data = training_data.to_numpy()
training_inputs = training_data[:, 0:500]
training_labels = training_data[:, 500]

for i in range(len(training_inputs)):
    Standardize(training_inputs, i)

input_tensor = torch.tensor(training_inputs)
label_tensor = torch.tensor(training_labels, dtype=torch.long)

model = PersistenceLandscapeNN()
learning_rate = 0.001
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for train_index, test_index in kf.split(input_tensor):
    X_train, X_test = input