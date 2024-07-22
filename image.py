import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

# Load the pre-trained model
model_path = r'C:\Users\arees\OneDrive\Desktop\Text_Generation_&_Summarizataion\hand-char-model.keras'
model = None
#this try and except will check if the model is loaded accurately
try:
    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Failed to load the model. Error: {e}")

# Preprocess image function
def preprocess(img):
    (h, w) = img.shape #extracts the height and width of image 64, 256
    final_img = np.ones([64, 256]) * 255  # generates a blank white image
    
    if w > 256:
        img = img[:, :256] #if image exceeds 256 width crops
    if h > 64:
        img = img[:64, :] #if image exceeds 64 height crops
    
    final_img[:h, :w] = img #it pastes the image on the white image
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

# Define alphabets and constants
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24 #the model can only process this much len of string
num_of_characters = len(alphabets) + 1 
num_of_timestamps = 64

# Convert label to numerical representation
# def label_to_num(label):
#     label_num = []
#     for ch in label:
#         label_num.append(alphabets.find(ch))
#     return np.array(label_num)

# Convert numerical representation to label
def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1: #special character under 0 will stop the process
            break
        else:
            ret += alphabets[ch] #adds the string ch to the ret
    return ret

# Extract text from image using the model
def extract_text_from_image(image_path, model):
    if model is None:
        raise ValueError("Model is not loaded. Please check the model path and loading mechanism.")
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read the image from {image_path}")
    
    image = preprocess(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=2) #(height, width, 1)
    image = np.expand_dims(image, axis=0) #(1, height, width, 1)
    
    preds = model.predict(image)
    decoded = tf.keras.backend.get_value(tf.keras.backend.ctc_decode(preds, input_length=np.ones(preds.shape[0]) * preds.shape[1], greedy=True)[0][0])
    return num_to_label(decoded[0])

# Test the extract_text_from_image function
if __name__ == "__main__":
    try:
        extracted_text = extract_text_from_image("temp_image.jpg", model) #stores the text from the image into the variable
        print("Extracted Text:", extracted_text)
    except Exception as e:
        print(f"Error during text extraction: {e}")
