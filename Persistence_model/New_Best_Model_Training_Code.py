import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from sklearn.model_selection import StratifiedKFold
from New_Best_Network import PersistenceLandscapeNN

def Standardize(data, index):
    arr = data[index]
    mean = np.mean(arr)
    std = np.std(arr)
    mean_array = np.full(len(arr), mean)
    result = (arr - mean_array) * (1 / std)
    data[index] = result

k = 30
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

training_data = pd.read_csv("../Preprocessing/Features/NoPrep_Training_Crop_L10B50.csv")
training_data = training_data.iloc[:, 1:]

num_glioma = len(training_data[training_data['label'] == 'glioma'])
num_meningioma = len(training_data[training_data['label'] == 'meningioma'])
num_notumor = len(training_data[training_data['label'] == 'notumor'])
num_pituitary = len(training_data[training_data['label'] == 'pituitary'])

num_per_class = min(num_glioma, num_meningioma, num_notumor, num_pituitary) - 10

training_data = training_data.to_numpy()

training_inputs = []
training_labels = []

validation_inputs = []
validation_labels = []

index = 0

while index < len(training_data):
    index_addition = 0
    
    if training_data[index, 0] == "glioma":
        index_addition = num_glioma
    elif training_data[index, 0] == "meningioma":
        index_addition = num_meningioma
    elif training_data[index, 0] == "notumor":
        index_addition = num_notumor
    else:
        index_addition = num_pituitary
    
    training_inputs += training_data[index:(index + num_per_class), 1:1001].tolist()
    
    training_labels += training_data[index:(index + num_per_class), 0].tolist()
    
    validation_inputs += training_data[(index + num_per_class):(index + index_addition), 1:1001].tolist()
    
    validation_labels += training_data[(index + num_per_class):(index + index_addition), 0].tolist()

    index += index_addition

training_inputs = np.array(training_inputs)
training_labels = np.array(training_labels)

validation_inputs = np.array(validation_inputs)
validation_labels = np.array(validation_labels)

# Shuffle the Training Set
training_shuffled_indices = np.random.permutation(len(training_inputs))
training_inputs = training_inputs[training_shuffled_indices]
training_labels = training_labels[training_shuffled_indices]

# Plot the histogram of the training label distribution
plt.figure(figsize=(8, 5))
plt.hist(training_labels, color="skyblue", edgecolor="black")
plt.title("Training Label Distribution")
plt.xlabel("Label")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("training_labels_dist.png")
plt.close()

print("\nTraining Labels Peek: ", training_labels[0:20])

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
learning_rate = 1e-3
loss_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
epochs = 15
training_loss_values = []
validation_loss_values = []
training_accuracy_values = []
validation_accuracy_values = []
current_patience_count = 0
tolerance = 1e-2
early_stopping_patience = 5
last_epoch = epochs
precision = dict({
    "glioma": [],
    "meningioma": [],
    "notumor": [],
    "pituitary": []
})

recall = dict({
    "glioma": [],
    "meningioma": [],
    "notumor": [],
    "pituitary": []
})

for epoch in range(epochs):
    print("\nTRAINING EPOCH " + str(epoch + 1) + " OF " + str(epochs))

    current_batch = 1
    training_loss = 0
    training_accuracy = 0
    true_positives = [0, 0, 0, 0]
    true_negatives = [0, 0, 0, 0]
    false_positives = [0, 0, 0, 0]
    false_negatives = [0, 0, 0, 0]
    
    for train_index, test_index in skf.split(training_input_tensor, training_label_tensor):
        print("-> Training with batch " + str(current_batch) + " of " + str(k))
        
        all_indices = train_index.tolist() + test_index.tolist()
        X, y = training_input_tensor[all_indices], training_label_tensor[all_indices]

        model.train()

        optimizer.zero_grad()
        predictions = model(X)
        loss = loss_criterion(predictions, y)
        training_loss += loss.item()
        loss.backward()
        optimizer.step()

        index = 0
        cnt = 0

        for vec in predictions:
            if torch.argmax(vec).item() == y[index]:
                cnt += 1
            index += 1

        training_accuracy += (cnt / len(y))

        current_batch += 1

    training_loss /= k
    training_accuracy /= k

    training_loss_values.append(training_loss)
    training_accuracy_values.append(training_accuracy)

    model.eval()

    predictions = model(validation_input_tensor)
    loss = loss_criterion(predictions, validation_label_tensor)

    cnt = 0
    index = 0
    
    for vec in predictions:
        if torch.argmax(vec).item() == validation_labels[index]:
            true_positives[validation_labels[index]] += 1
            for i in range(4):
                if i != validation_labels[index]:
                    true_negatives[i] += 1
            cnt += 1
        else:
            false_positives[torch.argmax(vec).item()] += 1
            false_negatives[validation_labels[index]] += 1
        index += 1

    if (len(validation_loss_values) > 0) and (loss.item() >= (validation_loss_values[len(validation_loss_values) - 1] - tolerance)):
        current_patience_count += 1
    else:
        current_patience_count = 0

    if current_patience_count == early_stopping_patience:
        print("\nEARLY STOPPING: Validation Loss Has Not Improved For " + str(early_stopping_patience) + " epochs!")
        last_epoch = epoch + 1
        validation_loss_values.append(loss.item())
        validation_accuracy_values.append(cnt / len(validation_labels))
        for target in label_map:
            if (true_positives[label_map[target]] > 0) and (false_positives[label_map[target]] > 0):
                precision[target].append(true_positives[label_map[target]] / (true_positives[label_map[target]] + false_positives[label_map[target]]))
            else:
                precision[target].append(0)
    
            if (true_positives[label_map[target]] > 0) and (false_negatives[label_map[target]] > 0):
                recall[target].append(true_positives[label_map[target]] / (true_positives[label_map[target]] + false_negatives[label_map[target]]))
            else:
                recall[target].append(0)
        break

    validation_loss_values.append(loss.item())
    validation_accuracy_values.append(cnt / len(validation_labels))
    
    for target in label_map:
        if (true_positives[label_map[target]] > 0) and (false_positives[label_map[target]] > 0):
            precision[target].append(true_positives[label_map[target]] / (true_positives[label_map[target]] + false_positives[label_map[target]]))
        else:
            precision[target].append(0)

        if (true_positives[label_map[target]] > 0) and (false_negatives[label_map[target]] > 0):
            recall[target].append(true_positives[label_map[target]] / (true_positives[label_map[target]] + false_negatives[label_map[target]]))
        else:
            recall[target].append(0)

epochs = list(range(1, last_epoch + 1))

plt.figure(figsize=(8, 5))
plt.plot(epochs, training_loss_values, label="Training Loss")
plt.plot(epochs, validation_loss_values, label="Validation Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.legend()
plt.savefig("current_loss.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs, training_accuracy_values, label="Training Accuracy")
plt.plot(epochs, validation_accuracy_values, label="Validation Accuracy")
plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("current_accuracy.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs, precision["glioma"], label="Glioma")
plt.plot(epochs, precision["meningioma"], label="Meningioma")
plt.plot(epochs, precision["notumor"], label="No Tumor")
plt.plot(epochs, precision["pituitary"], label="Pituitary")
plt.title("Precision Trend For Each Tumor Class")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.legend()
plt.savefig("current_precisions.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs, recall["glioma"], label="Glioma")
plt.plot(epochs, recall["meningioma"], label="Meningioma")
plt.plot(epochs, recall["notumor"], label="No Tumor")
plt.plot(epochs, recall["pituitary"], label="Pituitary")
plt.title("Recall Trend For Each Tumor Class")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.legend()
plt.savefig("current_recalls.png")
plt.close()

torch.save(model.state_dict(), "new_best_model.pth")
