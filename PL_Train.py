import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PL_Neural_Network import PersistenceLandscapeNN

"""
* Performs k-fold cross validation on the input data and labels.

* Parameters:
    k: The number of folds to split the data into.
    inputs: The input data.
    labels: The labels for the input data.

* Returns:
    A list of batches, where each
    batch is a list of tuples containing
    the input data and its corresponding label.
"""
def k_fold_cross_validation_split(k, inputs, labels):
    batches = []
    
    input_array = np.array(inputs)
    label_array = np.array(labels)
    batch_size = len(input_array) // k
    remaining_data = len(input_array) % k
    
    for i in range(k):
        batch = []
        
        print(f'\n-----------------Batch {i + 1} of {k}-----------------')
        
        for j in range(batch_size):
            index = i * batch_size + j
            batch.append((input_array[index], label_array[index]))
        
            if j == (batch_size // 4):
                print(f'Added {j + 1} data points to this batch.')
            elif j == (batch_size // 2):
                print(f'Added {j + 1} data points to this batch.')
            elif j == (batch_size * 3 // 4):
                print(f'Added {j + 1} data points to this batch.')
        
        print(f'Added {batch_size} data points to this batch.')
        batches.append(batch)
    
    for i in range(len(input_array) - remaining_data, len(input_array)):
        batches[len(batches) - 1].append((input_array[i], label_array[i]))
    
    if remaining_data > 0:
        print(f'Added {batch_size + remaining_data} data points to the last batch.')
    
    return batches

# The path to the CSV file with the persistence landscape features for the training images.
features_data_path = "brain_mri_dataset/training.csv"

# Load in the training features dataset.
features_data = pd.read_csv(features_data_path).to_numpy()

# Shuffle the training features dataset.
np.random.shuffle(features_data)

# Perform 80/20 split on the training and testing data features.
# !!REMOVE THIS CODE ONCE THE REAL TESTING IMAGE FEATURES HAVE BEEN EXTRACTED!!
training_features = features_data

# Split the training inputs into batches and extract the labels.
training_inputs = []
training_labels = []
k = 20

for i in range(len(training_features)):
    current_features = training_features[i]
    training_inputs.append(current_features[0:500])
    training_labels.append(current_features[500])

batches = k_fold_cross_validation_split(k, training_inputs, training_labels)

# Initialize the model and the hyper-parameters.
model = PersistenceLandscapeNN()
learning_rate = 0.001
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_values = []
training_accuracies = []

# print(batches[0])

# if (1 == 2):
# Train the network.
model.train()

for i in range(len(batches)):

    print("-------------------------------------TRAINING WITH BATCH " + str(i + 1) + "------------------------------------------------------")
    
    standardized_inputs = []
    indexed_labels = []

    for j in range(len(batches[i])):

        print("--> Standardizing Input " + str(j + 1) + " of " + str(len(batches[i])))
        
        INPUT, LABEL = batches[i][j]
        # total = np.sum(INPUT)
        mean = np.mean(INPUT)
        std = np.std(INPUT)
        mean_array = np.full(len(INPUT), mean)
        standardized_inputs.append((INPUT - mean_array) * (1 / std))
        print(standardized_inputs)
        indexed_labels.append(LABEL)

    X = np.array(standardized_inputs)
    print(standardized_inputs)
    break
    y = np.array(indexed_labels)

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.tensor(y, dtype=torch.long)

    optimizer.zero_grad()

    print("\n-> Forward Propagation In Progress...")
    predictions = model(X_tensor)
    print("-> Forward Propagation Complete!")

    print("-> Loss Computation In Progress...")
    loss_value = loss(predictions, y_tensor)
    print("-> Loss Computation Complete!\n")

    print("-> Back Propagation In Progress...")
    loss_value.backward()
    print("-> Back Propagation Complete!\n")
    
    optimizer.step()
    print("-> Took step towards local minimum!\n")

    predictions = predictions.detach().numpy()
    cnt = 0

    for i in range(len(predictions)):
        if np.argmax(predictions[i]) == y[i]:
            cnt += 1

    training_accuracies.append(cnt / len(predictions))
    loss_values.append(loss_value.item())
    
batches = list(range(1, k + 1))
plt.plot(batches, loss_values)
plt.savefig("Persistence Landscape NN Loss Values")

torch.save(model.state_dict(), 'PL_model_params.pth')