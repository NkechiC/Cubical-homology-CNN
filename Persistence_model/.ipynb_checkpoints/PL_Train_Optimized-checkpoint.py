import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from sklearn.model_selection import KFold
from PL_Neural_Network import PersistenceLandscapeNN

def Standardize(data, index):
    arr = data[index]
    mean = np.mean(arr)
    std = np.std(arr)
    mean_array = np.full(len(arr), mean)
    result = (arr - mean_array) * (1 / std)
    data[index] = result

k = 35
kf = KFold(n_splits=k, shuffle=True, random_state=42)

training_data = pd.read_csv("../Preprocessing/Extracted_Features.csv")
training_data = training_data.iloc[:, 1:]
training_data = training_data.to_numpy()
training_inputs = training_data[:, 1:201]
training_labels = training_data[:, 0]
label_map = dict({
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
})

for i in range(len(training_labels)):
    training_labels[i] = label_map[training_labels[i]]

for i in range(len(training_inputs)):
    Standardize(training_inputs, i)

training_inputs = training_inputs.astype(float)
training_labels = training_labels.astype(int)

input_tensor = torch.tensor(training_inputs, dtype=torch.float)
label_tensor = torch.tensor(training_labels, dtype=torch.long)

model = PersistenceLandscapeNN()
learning_rate = 0.002
loss_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.002)
epochs = 10
current_batch = 1
loss_values = []
accuracies = []

for train_index, test_index in kf.split(input_tensor):
    print("\nTRAINING WITH BATCH " + str(current_batch) + " OF " + str(k))

    loss_total = 0
    accuracy_total = 0

    for epoch in range(epochs):
        print("-> Epoch " + str(epoch + 1) + " of " + str(epochs))
        
        all_indices = train_index.tolist() + test_index.tolist()
        random.shuffle(all_indices)
        new_train_index = np.array(all_indices[0:len(all_indices) // 2])
        new_test_index = np.array(all_indices[len(all_indices) // 2:len(all_indices)])

        X_train, X_test = input_tensor[new_train_index], input_tensor[new_test_index]
        y_train, y_test = label_tensor[new_train_index], label_tensor[new_test_index]

        model.train()

        optimizer.zero_grad()
        predictions = model(X_train)
        loss = loss_criterion(predictions, y_train)
        loss_total += loss.item()
        loss.backward()
        optimizer.step()

        model.eval()

        test_predictions = model(X_test)
        test_predictions = test_predictions.detach().numpy()
        cnt = 0

        for i in range(len(test_predictions)):
            if np.argmax(test_predictions[i]) == y_test[i]:
                cnt += 1

        accuracy_total += (cnt / len(test_predictions))

    loss_values.append(loss_total / epochs)
    accuracies.append(accuracy_total / epochs)

    current_batch += 1

batches = list(range(1, k + 1))

plt.figure(figsize=(8, 5))
plt.plot(batches, loss_values)
plt.title("Training Loss")
plt.xlabel("Batch")
plt.ylabel("Loss Value")
plt.savefig("current_training_loss.png")

plt.figure(figsize=(8, 5))
plt.plot(batches, accuracies)
plt.title("Training Accuracy")
plt.xlabel("Batch")
plt.ylabel("Accuracy")
plt.savefig("current_training_accuracy.png")

torch.save(model.state_dict(), "current_model.pth")

    