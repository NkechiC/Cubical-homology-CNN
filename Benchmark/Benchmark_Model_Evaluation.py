import numpy as np
import random
import os
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from Best_Benchmark_NN import BenchmarkNN

# Initialize the model and set its parameters.
model = BenchmarkNN()
model.load_state_dict(torch.load("benchmark_model.pth"))

# Initialize a string containing the path to the testing images.
testing_path = "../../archive/Testing/"

# Initialize a list containing the four possible labels.
labels = ["glioma", "meningioma", "notumor", "pituitary"]

# Initialize a list contanining the frequency that each label was the output.
label_counts = [0, 0, 0, 0]

# Initialize a list containing the testing images.
testing_images = []

# Iterate through the labels.
for label in labels:

    # Set a variable to the directory of a label's testing images.
    label_dir = testing_path + label

    # Iterate over the images of the current label.
    for image in os.listdir(label_dir):

        # Skip the file if its is called .ipynb_checkpoints
        if image == ".ipynb_checkpoints":
            continue

        # Store the relative path to a specific image.
        image_path = label_dir + "/" + image

        # Add the relative path to the list of testing images.
        testing_images.append(image_path)

# Shuffle the list of testing images.
random.shuffle(testing_images)

# Predict the label of each image.
model.eval()

X = []
y = []

for i in range(len(testing_images)):
    img_path = testing_images[i]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    X.append(img)

    if 'glioma' in img_path:
        y.append(0)
    elif 'meningioma' in img_path:
        y.append(1)
    elif 'notumor' in img_path:
        y.append(2)
    else:
        y.append(3)

X = np.stack(X)
X_tensor = torch.from_numpy(X).float()
predictions = model(X_tensor)
predictions = predictions.detach().numpy()
cnt = 0

for i in range(len(predictions)):
    pred = np.argmax(predictions[i])
    label_counts[pred] += 1
    if pred == y[i]:
        cnt += 1

plt.bar(labels, label_counts)
plt.xlabel("Labels")
plt.ylabel("Count")
plt.savefig("benchmark_evaluation_labels.png")

print("Testing Accuracy: ", str(cnt / len(testing_images)))