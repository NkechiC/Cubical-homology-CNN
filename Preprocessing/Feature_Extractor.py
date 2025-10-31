import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import random
from PIL import Image
import cv2
import imutils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gtda.homology import CubicalPersistence
from gtda.diagrams import PersistenceLandscape

def CLAHE_persistence_landscape(clahe_skull):
    # Apply CLAHE
    # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    # clahe_skull = clahe.apply(gray_skull_stripped)
    
    # Normalize to [0, 1] for GTDA
    normalized_clahe_skull = clahe_skull / 255.0
    normalized_clahe_skull[normalized_clahe_skull < 0.02] = 0.5
    
    # Reshape for giotto-tda
    X_clahe_skull = normalized_clahe_skull[np.newaxis, :, :]  # Shape: (1, H, W)
    
    # Compute Cubical Persistence
    cp_skull = CubicalPersistence(homology_dimensions=(0, 1))
    cp_skull.fit(X_clahe_skull)
    diagrams_skull = cp_skull.fit_transform(X_clahe_skull)

    diagram = diagrams_skull[0]
    diagram_h0 = diagram[np.isclose(diagram[:, 2], 0)]
    diagram_h1 = diagram[np.isclose(diagram[:, 2], 1)]

    # Compute persistence landscape
    pl = PersistenceLandscape(n_layers=5, n_bins=50) 
    landscape_h0 = pl.fit_transform([diagram_h0])
    landscape_h1 = pl.fit_transform([diagram_h1])

    # Convert the persistence landscape to a feature vector
    features_h0 = landscape_h0[0].flatten()
    features_h1 = landscape_h1[0].flatten()

    return np.concatenate((features_h0, features_h1))

def Extract_Features(img_path):
    img = cv2.imread(img_path)
    return CLAHE_persistence_landscape(img)

def main():
    training_path = "../brain_mri_dataset/Training/"
    testing_path = "../brain_mri_dataset/Testing/"
    preprocessed_directories = [
        "glioma_preprocessed_comp",
        "meningioma_preprocessed_comp",
        "notumor_preprocessed_comp",
        "pituitary_preprocessed_comp"
    ]
    training_features_path = "../brain_mri_dataset/training.csv"
    testing_features_path = "../brain_mri_dataset/testing.csv"
    training_features = []
    testing_features = []
    headers = []

    for i in range(500):
        headers.append("feature_" + str(i))
    headers.append("output")

    training_features.append(headers)
    testing_features.append(headers)

    for directory in preprocessed_directories:
        training = training_path + directory
        testing = testing_path + directory

        # print("\nSTARTED EXTRACTING FEATURES FOR " + training + " images")

        # for image in os.listdir(training):
        #     if image == ".ipynb_checkpoints":
        #         continue

        #     image_path = training + "/" + image

        #     print("--> Extracting Features: ", image_path)

        #     training_features.append(
        #         Extract_Features(image_path).tolist()
        #     )

        #     if "glioma" in image_path:
        #         training_features[len(training_features) - 1].append(0)
        #     elif "meningioma" in image_path:
        #         training_features[len(training_features) - 1].append(1)
        #     elif "notumor" in image_path:
        #         training_features[len(training_features) - 1].append(2)
        #     else:
        #         training_features[len(training_features) - 1].append(3)

        print("\nSTARTED EXTRACTING FEATURES FOR " + testing + " images\n")
        
        for image in os.listdir(testing):

            if image == ".ipynb_checkpoints":
                continue

            image_path = testing + "/" + image

            print("--> Extracting Features: ", image_path)

            testing_features.append(
                Extract_Features(image_path).tolist()
            )

            if "glioma" in image_path:
                testing_features[len(testing_features) - 1].append(0)
            elif "meningioma" in image_path:
                testing_features[len(testing_features) - 1].append(1)
            elif "notumor" in image_path:
                testing_features[len(testing_features) - 1].append(2)
            else:
                testing_features[len(testing_features) - 1].append(3)

    # with open(training_features_path, mode="w", newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerows(training_features)

    with open(testing_features_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(testing_features)
    
            

if __name__ == "__main__":
    main()