# Cubical-homology-CNN

This repository contains code and models for preprocessing, feature extraction, and classification of MRI scans using both standard CNN-based and Topological Data Analysis (TDA)-based methods. The primary focus of this project is to apply Cubical Homology and Persistence Landscapes to extract robust topological features from brain MRI scans and classify them using a customized Neural Network.

## Overview

The models are trained to classify brain MRI scans into four categories:
- Glioma
- Meningioma
- No Tumor
- Pituitary

This project approaches the classification task using two separate pipelines:
1. **Topological Data Analysis (TDA) Pipeline**: Extracts topological features (Cubical Persistence Landscapes) from preprocessed MRI scans and uses a dense Neural Network for classification.
2. **Benchmark CNN Pipeline**: Trains a standard Convolutional Neural Network (CNN) directly on the MRI images to compare performance and robustness with the TDA-based approach.

## Repository Structure

- **`Preprocessing/`**
  Contains scripts to preprocess the raw MRI images and extract topological features.
  - **Image Cropping**: Isolates the brain structure by removing excess background.
  - **Skull-stripping**: Removes non-brain tissue using a MATLAB-based algorithm.
  - **CLAHE**: Contrast Limited Adaptive Histogram Equalization to improve image contrast.
  - **Feature Extraction**: Scripts like `Feature_Extractor.py` and `NoPrep_Feature_Extractor.py` compute Cubical Persistence and generate Persistence Landscapes using `giotto-tda`.

- **`Persistence_model/`**
  Contains the PyTorch implementation of the Neural Network designed to train on the extracted topological features (Persistence Landscapes). It includes training scripts (e.g., `Complete_script.py`, `PL_Train_Optimized.py`), cross-validation procedures, and model evaluation metrics.

- **`Benchmark/`**
  Contains the baseline CNN architecture (`Best_Benchmark_NN.py`) and training scripts (`Benchmark_Training.py`). The CNN is trained directly on the resized/cropped MRI images to serve as a benchmark against the TDA pipeline.

- **`Tradition_ML/`**
  Contains Jupyter Notebooks and scripts exploring traditional machine learning approaches and baseline reproductions (e.g., MNIST reproduction).

## Key Dependencies

- **Deep Learning**: `torch`, `torchvision`
- **Topological Data Analysis**: `giotto-tda` (`gtda.homology`, `gtda.diagrams`)
- **Computer Vision**: `opencv-python` (`cv2`), `Pillow`, `imutils`
- **Data & Visualization**: `numpy`, `pandas`, `matplotlib`, `scikit-learn`
- **Other**: MATLAB Engine API for Python (specifically for the skull-stripping module)
