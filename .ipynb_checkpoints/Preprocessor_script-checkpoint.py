################################################################################################################################################################
########################################################## Declaring functions to perform each pre-processing component ########################################
################################################################################################################################################################

"""
Finds the extreme points on the image and crops the rectangular out of them
"""
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

def crop_and_resize(image_path):       
    # Load and preprocess the image
    IMG_SIZE = 256
    image = Image.open(image_path).convert('L')
    shape = tuple(image.size)

    image_np_array = np.array(image)

    cropped_skull = crop_img(image_np_array, shape)
    resized = cv2.resize(cropped_skull, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    return resized

def save_cropped_image(img, original_path):
    folder = "CROPPED_AND_RESIZED"
    filename = os.path.basename(original_path)

    new_path = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)

    cv2.imwrite(new_path, img)

    return new_path

def save_skull_stripped_image(img, original_path):
    folder = "SKULL_STRIPPED_IMAGES"
    filename = os.path.basename(original_path)

    new_path = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)

    cv2.imwrite(new_path, img)

    return new_path

def skull_stripping(img_path):
    # Use current directory where notebook and .m/.mat files are located
    eng.cd(os.path.abspath("Skull-stripping"), nargout=0)
    img_path = os.path.abspath(img_path)
    model_path = os.path.abspath("Skull-stripping/NIVE.mat")
    
    result = eng.nive_extract_brain(img_path, model_path)
    
    # Convert MATLAB array to NumPy
    # result is a 3D array in column-major (Fortran-style)
    shape = tuple(result.size)  # (height, width, channels)
    skull_np_array = np.array(result._data, dtype=np.uint8).reshape(shape, order='F')

    cropped_skull = crop_img(skull_np_array, shape)
    resized_skull = cv2.resize(cropped_skull, (200, 200), interpolation=cv2.INTER_AREA)

    return resized_skull

def CLAHE_persistence_landscape(gray_skull_stripped):
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    clahe_skull = clahe.apply(gray_skull_stripped)
    
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
    
################################################################################################################################################################
#################################################################### Defining the Preprocessor class ###########################################################
################################################################################################################################################################

class Preprocessor:
    def __init__(self, batch_length):
        self.cnt = 1
        self.batch_length = batch_length
    
    def extract_landscapes(self, image_path):
        cropped_img = crop_and_resize(image_path)
        cropped_img_path = save_cropped_image(cropped_img, image_path)
        skull = skull_stripping(cropped_img_path)
        skull = cv2.resize(skull, (200, 200), interpolation=cv2.INTER_AREA)
        skull_stripped_img_path = save_skull_stripped_image(skull, image_path)
        landscape = CLAHE_persistence_landscape(skull)
        print("-> [" + str(self.cnt) + "/" + str(self.batch_length) + "] EXTRACTED PERSISTENCE LANDSCAPE FOR " + image_path)
        self.cnt += 1
        return landscape
        
################################################################################################################################################################
#################################################################### Storing the Training and Testing Data #####################################################
################################################################################################################################################################

# Initialize a string containing the path to the training images.
training_path = "brain_mri_dataset/Training/"

# Initialize a string containing the path to the testing images.
testing_path = "brain_mri_dataset/Testing/"

# Initialize a list containing the four possible labels.
labels = ["glioma", "meningioma", "notumor", "pituitary"]

# Initialize a list to contain the training images.
training_images = []

# Initialize a list containing the testing images.
testing_images = []

# Iterate through the labels.
for label in labels:

    # Set a variable to the directory of a label's training images.
    label_dir = training_path + label

    # Iterate over the images of the current label.
    for image in os.listdir(label_dir):

        # Skip the file if its is called .ipynb_checkpoints
        if image == ".ipynb_checkpoints":
            continue

        # Store the relative path to a specific image.
        image_path = label_dir + "/" + image

        # Add the relative path to the list of training images.
        training_images.append(image_path)

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

################################################################################################################################################################################################################### Preprocessing the training data using the Preprocessor class ###############################################
################################################################################################################################################################

training_preprocessor = Preprocessor(len(training_images))
training_csv_contents = []
headers = []

for i in range(1, 1001):
    headers.append("feature_" + str(i))

headers.append("output")

training_csv_contents.append(headers)

for i in range(len(training_images)):
    landscape = training_preprocessor.extract_landscapes(training_images[i])
    landscape = landscape.tolist()

    if "glioma" in training_images[i]:
        landscape.append(0)
    elif "meningioma" in training_images[i]:
        landscape.append(1)
    elif "notumor" in training_images[i]:
        landscape.append(2)
    else:
        landscape.append(3)

    training_csv_contents.append(landscape)

################################################################################################################################################################################################################### Preprocessing the testing data using the Preprocessor class ################################################
################################################################################################################################################################

testing_preprocessor = Preprocessor(len(testing_images))
testing_csv_contents = []
headers = []

for i in range(1, 1001):
    headers.append("feature_" + str(i))

headers.append("output")

testing_csv_contents.append(headers)

for i in range(len(training_images)):
    landscape = training_preprocessor.extract_landscapes(training_images[i])
    landscape = landscape.tolist()

    if "glioma" in training_images[i]:
        landscape.append(0)
    elif "meningioma" in training_images[i]:
        landscape.append(1)
    elif "notumor" in training_images[i]:
        landscape.append(2)
    else:
        landscape.append(3)

    testing_csv_contents.append(landscape)

################################################################################################################################################################################################################### Save the pre-processed training and testing data to a .csv file ############################################
################################################################################################################################################################

with open("training_data.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(training_csv_contents)

with open("testing_data.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(testing_csv_contents)