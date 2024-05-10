from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import keras
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)
# Load your saved model (replace with your filename)
model = tf.keras.models.load_model("eye_disease_classifier.h5")

from tensorflow.keras.preprocessing.image import img_to_array, load_img

from tensorflow.keras.applications.vgg16 import preprocess_input

def preprocess_image(image):
  """Preprocesses an uploaded image for the model.

  Args:
      image: A file object (assumed to be a Flask FileStorage object) containing the uploaded image.

  Returns:
      A NumPy array representing the preprocessed image, or None if an error occurs.
  """

  try:
    # Load the image from the file object
    img = load_img(image, target_size=(224, 224))  # Resize to target size

    # Convert the image to a NumPy array
    x = img_to_array(img)

    # Convert to float32 for normalization
    x = x.astype('float32')

    # Normalize pixel values between 0 and 1 (assuming VGG16 expects this)
    x /= 255.0

    # Expand dimensions for batch processing (add a dimension as the first dimension)
    x = np.expand_dims(x, axis=0)

    return x

  except Exception as e:
    print(f"Error preprocessing image: {e}")
    return None



@app.route("/classify", methods=["POST"])
def classify_image():
    # Get the uploaded image from the request
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]

    import time
    filename = f"uploads/image_{time.time()}.jpg"

    image.save(filename)

    # Preprocess the image
    preprocessed_image = preprocess_image(filename)

    # Make predictions using the model
    prediction = model.predict(preprocessed_image)

    # Convert prediction to class label (assuming you have class labels)
    predicted_class = int(np.argmax(prediction))  # Get index of highest probability class

    if preprocessed_image is not None:
        prediction = model.predict(preprocessed_image)
        print(prediction)
        return jsonify({
            "class":  predicted_class,
            "prediction": prediction.tolist(),
            "status": str(200)  # Convert integer status code to string
        })
    else:
        return jsonify({"error": "Error preprocessing image"}), 400

if __name__ == "__main__":
    app.run(debug=True)  # Set debug=True for automatic reloading during development

