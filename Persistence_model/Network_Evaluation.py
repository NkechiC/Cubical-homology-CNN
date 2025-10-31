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

model = PersistenceLandscapeNN()
model.load_state_dict(torch.load("current_model.pth"))
model.eval()

testing_data = pd.read_csv("../Preprocessing/Extracted_Features_Testing.csv")
testing_data = testing_data.iloc[:, 1:]
testing_data = testing_data.to_numpy()

np.random.shuffle(testing_data)

testing_inputs = testing_data[:, 1:201]
testing_labels = testing_data[:, 0]
label_map = dict({
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
})

for i in range(len(testing_labels)):
    testing_labels[i] = label_map[testing_labels[i]]

for i in range(len(testing_inputs)):
    Standardize(testing_inputs, i)

testing_inputs = testing_inputs.astype(float)
testing_labels = testing_labels.astype(int)
loss_criterion = nn.CrossEntropyLoss()
correct_count = 0
images = []
loss_values = []
label_counts = [0, 0, 0, 0]

for i in range(len(testing_inputs)):
    print("\nTESTING WITH IMAGE " + str(i + 1) + " OF " + str(len(testing_inputs)))

    images.append(i + 1)

    X = torch.tensor([testing_inputs[i]], dtype=torch.float)
    y = torch.tensor([testing_labels[i]], dtype=torch.long)

    prediction = model(X)

    print("-> Prediction Class: " + str(np.argmax(prediction.detach().numpy()[0])))
    print("-> Target Class: " + str(testing_labels[i]))
    print("-> Prediction Vector: " + str(prediction.detach().numpy()[0]))
    
    loss = loss_criterion(prediction, y)
    loss_values.append(loss.item())

    print("-> Loss Value (Cross Entropy): " + str(loss.item()))

    if np.argmax(prediction.detach().numpy()[0]) == testing_labels[i]:
        correct_count += 1

    label_counts[np.argmax(prediction.detach().numpy()[0])] += 1

plt.figure(figsize=(8, 5))
plt.hist(loss_values, bins=50, color="skyblue", edgecolor="black")
plt.title("Evaluation Loss Distribution")
plt.xlabel("Loss Value")
plt.ylabel("Frequency")
plt.savefig("testing_loss.png")

print("\nEvaluation Accuracy: " + str(correct_count / len(testing_inputs)))