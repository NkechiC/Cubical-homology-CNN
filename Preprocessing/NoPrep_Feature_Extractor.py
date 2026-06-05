import numpy as np
import pandas as pd
import csv
import os
import cv2
import imutils
from gtda.homology import CubicalPersistence
from gtda.diagrams import PersistenceLandscape

def crop_img(gray, size):
    if len(size) > 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    
    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    new_img = gray[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    
    return new_img

def extract_cubical_features(img_path):
    img = cv2.imread(img_path)
    
    # Apply Crop
    cropped_img = crop_img(img, img.shape)
    
    # Normalize to [0, 1] for GTDA
    normalized_img = cropped_img / 255.0
    normalized_img[normalized_img < 0.02] = 0.5
    
    # Reshape for giotto-tda
    X_img = normalized_img[np.newaxis, :, :]  # Shape: (1, H, W)
    
    # Compute Cubical Persistence
    cp = CubicalPersistence(homology_dimensions=(0, 1))
    cp.fit(X_img)
    diagrams = cp.fit_transform(X_img)

    diagram = diagrams[0]
    diagram_h0 = diagram[np.isclose(diagram[:, 2], 0)]
    diagram_h1 = diagram[np.isclose(diagram[:, 2], 1)]

    # Compute persistence landscape: n_layers=10, n_bins=50
    pl = PersistenceLandscape(n_layers=10, n_bins=50) 
    landscape_h0 = pl.fit_transform([diagram_h0])
    landscape_h1 = pl.fit_transform([diagram_h1])

    # Convert the persistence landscape to a feature vector
    features_h0 = landscape_h0[0].flatten()
    features_h1 = landscape_h1[0].flatten()

    return np.concatenate((features_h0, features_h1))

def main():
    training_path = "../brain_mri_dataset/Training/"
    testing_path = "../brain_mri_dataset/Testing/"
    
    directories = [
        "glioma",
        "meningioma",
        "notumor",
        "pituitary"
    ]
    
    training_features_path = "../brain_mri_dataset/NoPrep_Training_Crop_L10B50.csv"
    testing_features_path = "../brain_mri_dataset/NoPrep_Testing_Crop_L10B50.csv"
    
    training_features = []
    testing_features = []
    headers = []

    # 10 layers * 50 bins = 500 features per homology dimension (H0 and H1). Total = 1000 features.
    for i in range(1000):
        headers.append("feature_" + str(i))
    headers.append("output")

    training_features.append(headers)
    testing_features.append(headers)

    # Note: In Feature_Extractor.py, the training extraction loop was commented out. 
    # I am uncommenting it here so it actually creates the training CSV.
    for directory in directories:
        training = training_path + directory
        testing = testing_path + directory

        print("\nSTARTED EXTRACTING FEATURES FOR " + training + " images\n")
        
        # Training extraction
        if os.path.exists(training):
            for image in os.listdir(training):
                if image == ".ipynb_checkpoints":
                    continue

                image_path = training + "/" + image
                print("--> Extracting Features: ", image_path)

                training_features.append(
                    extract_cubical_features(image_path).tolist()
                )

                if "glioma" in image_path:
                    training_features[-1].append(0)
                elif "meningioma" in image_path:
                    training_features[-1].append(1)
                elif "notumor" in image_path:
                    training_features[-1].append(2)
                else:
                    training_features[-1].append(3)

        print("\nSTARTED EXTRACTING FEATURES FOR " + testing + " images\n")
        
        # Testing extraction
        if os.path.exists(testing):
            for image in os.listdir(testing):
                if image == ".ipynb_checkpoints":
                    continue

                image_path = testing + "/" + image
                print("--> Extracting Features: ", image_path)

                testing_features.append(
                    extract_cubical_features(image_path).tolist()
                )

                if "glioma" in image_path:
                    testing_features[-1].append(0)
                elif "meningioma" in image_path:
                    testing_features[-1].append(1)
                elif "notumor" in image_path:
                    testing_features[-1].append(2)
                else:
                    testing_features[-1].append(3)

    print("\nSaving Training Features CSV...")
    with open(training_features_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(training_features)

    print("\nSaving Testing Features CSV...")
    with open(testing_features_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(testing_features)
        
    print("\nFinished successfully!")

if __name__ == "__main__":
    main()
