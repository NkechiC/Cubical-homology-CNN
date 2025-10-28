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

k = 20
kf = KFold(n_splits=k, shuffle=True, random_state=42)

training_data = pd.read_csv("../brain_mri_dataset/training.csv")
training_data = training_data.to_numpy()
training_inputs = training_data[:, 0:500]
training_labels = training_data[:, 500]

for i in range(len(training_inputs)):
    Standardize(training_inputs, i)

input_tensor = torch.tensor(training_inputs, dtype=torch.float)
label_tensor = torch.tensor(training_labels, dtype=torch.long)

model = PersistenceLandscapeNN()
learning_rate = 0.001
loss_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 5
current_batch = 1
loss_values = []
accuracies = []

for train_index, test_index in kf.split(input_tensor):
    print("\nTRAINING WITH BATCH " + str(current_batch) + " OF " + str(k))
    
    X_train, X_test = input_tensor[train_index], input_tensor[test_index]
    y_train, y_test = label_tensor[train_index], label_tensor[test_index]
    loss_total = 0

    model.train()
    
    for epoch in range(epochs):
        print("-> Epoch " + str(epoch + 1) + " of " + str(epochs))
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = loss_criterion(predictions, y_train)
        loss_total += loss.item()
        loss.backward()
        optimizer.step()

    loss_values.append(loss_total / epochs)

    print("-> Evaluating With Testing Subset")

    model.eval()

    test_predictions = model(X_test)
    test_predictions = test_predictions.detach().numpy()
    cnt = 0

    for i in range(len(test_predictions)):
        if np.argmax(test_predictions[i]) == y_test[i]:
            cnt += 1

    accuracies.append(cnt / len(test_predictions))

    current_batch += 1

batches = list(range(1, k + 1))

plt.figure(figsize=(8, 5))
plt.plot(batches, loss_values)
plt.title("Training Loss")
plt.xlabel("Batch")
plt.ylabel("Loss Value")
plt.savefig("training_loss.png")

plt.figure(figsize=(8, 5))
plt.plot(batches, accuracies)
plt.title("Training Accuracy")
plt.xlabel("Batch")
plt.ylabel("Accuracy")
plt.savefig("training_accuracy.png")

torch.save(model.state_dict(), "current_model.pth")

    