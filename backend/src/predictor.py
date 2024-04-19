from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import io

# Load the trained model
model = load_model('./model/traffic_sign_classifier.h5')

def predict(image_bytes):
    # Convert bytes to a numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode the numpy array as an image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize the image to match the model's expected input size
    image = cv2.resize(image, (30, 30))

    # Expand dimensions to match the model's expected input shape
    image = np.expand_dims(image, axis=0)

    # Normalize the image data to the range [0, 1]
    image = image / 255

    # Perform the prediction
    prediction = model.predict(image)
    
    # Return the prediction
    return prediction[0]