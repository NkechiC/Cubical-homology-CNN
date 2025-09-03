#!/usr/bin/env python
# coding: utf-8

"""
Preprocess MRI images:
1. Crop images
2. Run skull stripping with MATLAB
3. Apply CLAHE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import imutils
import matlab.engine
import random
from gtda.homology import CubicalPersistence
from gtda.diagrams import PersistenceLandscape
import plotly.express as px


def crop_img(img):
    """Finds the extreme points on the image and crops the rectangle."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0

    new_img = img[
        extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
        extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS
    ].copy()
    
    return new_img


def complete_preprocessing(image_path, eng):
    """Complete preprocessing for a given image path."""
    eng.cd(os.getcwd(), nargout=0)
    model_path = os.path.abspath("NIVE.mat")

    result = eng.nive_extract_brain(image_path, model_path)
    shape = tuple(result.size)
    skull_np_array = np.array(result._data, dtype=np.uint8).reshape(shape, order='F')
    
    cropped_skull = crop_img(skull_np_array)
    resized_skull = cv2.resize(cropped_skull, (200, 200), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized_skull, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    clahe_skull = clahe.apply(gray)
    
    return Image.fromarray(clahe_skull)


if __name__ == "__main__":
    eng = matlab.engine.start_matlab()

    # # First stage
    # input_dir = "MRI/Training/glioma"
    # output_dir = "MRI/Training/glioma_preprocessed"
    # os.makedirs(output_dir, exist_ok=True)

    # IMG_SIZE = 256
    # for filename in os.listdir(input_dir):
    #     if filename.lower().endswith(".jpg"):
    #         image_path = os.path.join(input_dir, filename)
    #         image = cv2.imread(image_path)
    #         cropped = crop_img(image)
    #         resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
    #         output_path = os.path.join(output_dir, filename)
    #         cv2.imwrite(output_path, resized)

    # Second stage
    new_input_dir = "MRI/Training/meningioma_preprocessed"
    new_output_dir = "MRI/Training/meningioma_preprocessed_comp"
    os.makedirs(new_output_dir, exist_ok=True)

    sorted_filenames = sorted([f for f in os.listdir(new_input_dir) if f.lower().endswith(".jpg")])
    for filename in sorted_filenames:
        image_path = os.path.join(new_input_dir, filename)
        try:
            processed_img = complete_preprocessing(image_path, eng)
            output_path = os.path.join(new_output_dir, filename)
            processed_img.save(output_path)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    eng.quit()
