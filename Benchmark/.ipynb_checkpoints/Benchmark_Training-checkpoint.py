import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
from sklearn.model_selection import KFold
from Best_Benchmark_NN import BenchmarkNN

def ReadImage(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img
    

k = 50
kf = KFold(n_splits=k, shuffle=True, random_state=42)

training_path = "../brain_mri_dataset/Training/"
labels = dict({
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
})

training_images = []
training_labels = []

for label in labels:
    label_dir = training_path + label

    for image in os.listdir(label_dir):
        if image == ".ipynb_checkpoints":
            continue
        image_path = label_dir + "/" + image
        training_images.append(image_path)
        training_labels.append(labels[label])

combined = list(zip(training_images, training_labels))
random.shuffle(combined)
images_shuffled, labels_shuffled = zip(*combined)
training_images = list(images_shuffled)
training_labeld = list(labels_shuffled)

learning_rate = 1e-10
model = BenchmarkNN()
loss_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 3
loss_values = []
img_nums = []

model.train()

for i in range(len(training_images)):
    print("\nTRAINING WITH IMAGE " + str(i + 1) + " OF " + str(len(training_images)))

    img_nums.append(i + 1)
    
    X = np.stack([ReadImage(training_images[i])])
    X = torch.from_numpy(X).float()
    
    y = torch.tensor([training_labels[i]], dtype=torch.long)

    loss_total = 0

    for epoch in range(epochs):
        print("-> Epoch " + str(epoch + 1) + " OF " + str(epochs))
        optimizer.zero_grad()

        print("    [1/4] Forward Propagation")
        predictions = model(X)

        print("    [2/4] Loss Computation")
        loss = loss_criterion(predictions, y)
        loss_total += loss.item()

        print("    [3/4] Back Propagation")
        loss.backward()

        print("    [4/4] Parameter Update")
        optimizer.step()

    loss_values.append(loss_total / epochs)

plt.figure(figsize=(8, 5))
plt.plot(img_nums, loss_values)
plt.title("Training Loss")
plt.xlabel("Image")
plt.ylabel("Loss Value")
plt.savefig("current_training_loss.png")