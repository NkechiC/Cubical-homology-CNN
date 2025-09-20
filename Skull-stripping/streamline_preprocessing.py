import numpy as np
import cv2
import imutils
import os
from PIL import Image
import matlab.engine


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

    new_img = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]].copy()
    return new_img


def complete_preprocessing(image_path, eng=None, skull_strip=True, apply_clahe=True):
    """Preprocess image with options for skull-stripping and CLAHE."""

    if skull_strip:
        # Skull stripping 
        eng.cd(os.getcwd(), nargout=0)
        model_path = os.path.abspath("NIVE.mat")
        result = eng.nive_extract_brain(image_path, model_path)
        shape = tuple(result.size)
        skull_np_array = np.array(result._data, dtype=np.uint8).reshape(shape, order="F")
        img = skull_np_array

        # Crop + resize 
        cropped = crop_img(img)
        resized = cv2.resize(cropped, (200, 200), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    else:
        # If no skull stripping: just read and grayscale
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE or plain grayscale
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        final_img = clahe.apply(gray)
    else:
        final_img = gray

    return Image.fromarray(final_img)

if __name__ == "__main__":
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
    
    new_input_dir = "MRI/Training/pituitary_preprocessed"
    sorted_filenames = sorted([f for f in os.listdir(new_input_dir) if f.lower().endswith(".jpg")])

    # Flags
    use_skull_strip = True
    use_clahe = True

    eng = None
    if use_skull_strip:
        eng = matlab.engine.start_matlab()

    cache = {} 

    # Process first 5 images
    for filename in sorted_filenames[:5]:
        image_path = os.path.join(new_input_dir, filename)

        try:
            processed_img = complete_preprocessing(image_path, eng=eng, skull_strip=use_skull_strip, apply_clahe=use_clahe)
            cache[filename] = processed_img
            print(f"Processed and cached: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Show the cached images
    for name, img in cache.items():
        img.show(title=name) 

    # Quit MATLAB only if it was started
    if eng is not None:
        eng.quit()

    print(f"\nCached {len(cache)} images in memory.")
