import numpy as np
import cv2
import imutils
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from gtda.homology import CubicalPersistence
from gtda.diagrams import PersistenceLandscape, features
from scipy.ndimage import convolve


def crop_img(img):
	"""
	Finds the extreme points on the image and crops the rectangular out of them
	"""
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
	new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
	
	return new_img


def compute_persistence_and_critical_points(filepath, image_size=200):
    """Crop image, compute persistence diagram, and extract critical filtration values."""
    # Load + preprocess
    image = cv2.imread(filepath)
    cropped = crop_img(image)
    resized = cv2.resize(cropped, (image_size, image_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0  # normalize to [0,1]

    # Shape for giotto-tda
    X = gray[np.newaxis, :, :]

    # Compute cubical persistence
    cp = CubicalPersistence(homology_dimensions=[0, 1])
    diagrams = cp.fit_transform(X)
    diagram = diagrams[0]

    # Separate by homology dimension
    diagram_h0 = diagram[np.isclose(diagram[:, 2], 0)]
    diagram_h1 = diagram[np.isclose(diagram[:, 2], 1)]

    # Extract unique critical filtration values 
    critical_points_h0 = np.unique(np.concatenate([diagram_h0[:, 0], diagram_h0[:, 1]]))
    critical_points_h1 = np.unique(np.concatenate([diagram_h1[:, 0], diagram_h1[:, 1]]))

    return diagram, gray, critical_points_h0, critical_points_h1


def neighbor_avg_diff(gray, t_i):
    """
    For each filtration value t_i, compute:
    1. Average of neighbors of all pixels with that value (excluding itself)
    2. Absolute difference between pixel value and neighbor average
    """
    weight = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]])

    # Sum of neighbors
    neighbor_sum = convolve(gray, weight, mode='constant', cval=0.0)
    # Count of neighbors (for boundary pixels)
    neighbor_count = convolve(np.ones_like(gray), weight, mode='constant', cval=0.0)

    avg_neighbors = []
    abs_diff = []

    for val in t_i:
        mask = np.isclose(gray, val, atol=1e-6) 
        if np.sum(mask) == 0:
            avg_neighbors.append(0)
            abs_diff.append(0)
            continue

        # Average of neighbors for pixels with intensity val
        neighbor_values = neighbor_sum[mask] / neighbor_count[mask]
        avg_val = np.mean(neighbor_values)
        avg_neighbors.append(avg_val)
        abs_diff.append(np.abs(avg_val - val))

    return np.array(avg_neighbors), np.array(abs_diff)

if __name__ == "__main__":
    image_path = "MRI/Training/glioma/Tr-gl_0050.jpg"
    diagram, gray, crit_h0, crit_h1 = compute_persistence_and_critical_points(image_path)

    avg_h0, diff_h0 = neighbor_avg_diff(gray, crit_h0)
    avg_h1, diff_h1 = neighbor_avg_diff(gray, crit_h1)

    #print("\nCritical filtration points (H0):", crit_h0)
    # print("Neighbor averages (H0):", avg_h0)
    # print("Neighbor diffs (H0):", diff_h0)
    # print("Length of critical points (H0):", len(crit_h0))
    # print(crit_h0)
    # print(crit_h1)

    #print("\nCritical filtration points (H1):", crit_h1)
    # print("Neighbor averages (H1):", avg_h1)
    # print("Neighbor diffs (H1):", diff_h1)
    # print("Length of critical points (H1):", len(crit_h1))

    # Plot persistence diagram
    # plt.scatter(diagram[:, 0], diagram[:, 1], c=diagram[:, 2], cmap="coolwarm")
    # plt.plot([0, 1], [0, 1])
    # plt.xlabel("Birth")
    # plt.ylabel("Death")
    # plt.title("Cubical Persistence Diagram")
    # plt.show()

    # plt.imshow(gray, cmap='gray')  
    # for i in crit_h0:
    #     coordinates = np.argwhere(np.isclose(gray, i, atol=1e-6))
    #     if len(coordinates) > 0:
    #         y,x = coordinates.T
    #         plt.scatter(x,y, s=10, c='red')
    # plt.title("Critical Filtration Points (H0)")
    # plt.axis('off')
    # plt.show()