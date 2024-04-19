from flask import Flask, request
from flask_cors import CORS
from predictor import predict

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict_route():
    # Check if any image was posted
    if 'image' not in request.files:
        return 'No image provided', 400

    # Get the images
    image_files = request.files.getlist('image')

    predictions = []
    for image_file in image_files:
        image = image_file.read()

        # Perform the prediction using your model
        prediction = predict(image)

        # Append the prediction to the list of predictions
        predictions.append(prediction.tolist())  # Convert numpy array to list

    # Return the predictions as a JSON response
    return {'predictions': predictions}, 200

if __name__ == '__main__':
    app.run()