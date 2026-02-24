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
np.random.shuffle(training_data)

training_inputs = training_data[0:4569, 1:201]
training_labels = training_data[0:4569, 0]

validation_inputs = training_data[4569:5712, 1:201]
validation_labels = training_data[4569:5712, 0]

label_map = dict({
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
})

for i in range(len(training_labels)):
    training_labels[i] = label_map[training_labels[i]]

    if i < len(validation_labels):
        validation_labels[i] = label_map[validation_labels[i]]

for i in range(len(training_inputs)):
    Standardize(training_inputs, i)

for i in range(len(validation_inputs)):
    Standardize(validation_inputs, i)

training_inputs = training_inputs.astype(float)
training_labels = training_labels.astype(int)

validation_inputs = validation_inputs.astype(float)
validation_labels = validation_labels.astype(int)

training_input_tensor = torch.tensor(training_inputs, dtype=torch.float)
training_label_tensor = torch.tensor(training_labels, dtype=torch.long)

validation_input_tensor = torch.tensor(validation_inputs, dtype=torch.float)
validation_label_tensor = torch.tensor(validation_labels, dtype=torch.long)

model = PersistenceLandscapeNN()
learning_rate = 1e-4
loss_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
epochs = 10
training_loss_values = []
validation_loss_values = []
training_accuracy_values = []
validation_accuracy_values = []
current_batch = 1

for train_index, test_index in kf.split(training_input_tensor):
    print("\nTRAINING WITH BATCH " + str(current_batch) + " OF " + str(k))

    training_loss = 0
    training_accuracy = 0
    all_indices = train_index.tolist() + test_index.tolist()

    X_train = training_input_tensor[all_indices]
    y_train = training_label_tensor[all_indices]

    model.train()
    
    for epoch in range(epochs):
        print("-> Epoch " + str(epoch + 1) + " of " + str(epochs))
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = loss_criterion(predictions, y_train)
        training_loss += loss.item()
        loss.backward()
        optimizer.step()

        index = 0
        cnt = 0

        for vec in predictions:
            if torch.argmax(vec).item() == training_labels[index]:
                cnt += 1
            index += 1

        training_accuracy += (cnt / len(training_labels))

    training_loss /= epochs
    training_accuracy /= epochs

    training_loss_values.append(training_loss)
    training_accuracy_values.append(training_accuracy)

    model.eval()

    predictions = model(validation_input_tensor)
    loss = loss_criterion(predictions, validation_label_tensor)
    validation_loss_values.append(loss.item())

    cnt = 0
    index = 0
    
    for vec in predictions:
        if torch.argmax(vec).item() == validation_labels[index]:
            cnt += 1
        index += 1

    validation_accuracy_values.append(cnt / len(validation_labels))
    
    current_batch += 1

batches = list(range(1, k + 1))

plt.figure(figsize=(8, 5))
plt.plot(batches, training_loss_values, label="Training Loss")
plt.plot(batches, validation_loss_values, label="Validation Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Batch")
plt.ylabel("Loss Value")
plt.legend()
plt.savefig("current_loss_1.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(batches, training_accuracy_values, label="Training Accuracy")
plt.plot(batches, validation_accuracy_values, label="Validation Accuracy")
plt.title("Training & Validation Accuracy")
plt.xlabel("Batch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("current_accuracy_1.png")
plt.close()


torch.save(model.state_dict(), "current_model_1.pth")

    
