from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import string
import os

app = Flask(__name__)
loaded_model = tf.keras.models.load_model("hand_signal_detection_model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400
    
    image = request.files['image']
    image_path = "temp.jpg"  # Path to temporarily save the image
    image.save(image_path)

    # Load and preprocess the image
    image_width, image_height = 200, 200
    uploaded_image = load_img(image_path, target_size=(image_width, image_height))
    array_of_uploaded_image = img_to_array(uploaded_image)
    array_of_uploaded_image = array_of_uploaded_image.reshape((1,) + array_of_uploaded_image.shape)
    array_of_uploaded_image = array_of_uploaded_image.astype("float32") / 255

    # Make predictions
    predictions = loaded_model.predict(array_of_uploaded_image)
    predicted_label_index = predictions.argmax()

    # Get the predicted label based on the index
    class_labels = list(string.ascii_uppercase) + ["nothing", "space"]
    if predicted_label_index >= len(class_labels):
        predicted_label = "nothing"  # Default label for out-of-range index
    else:
        predicted_label = class_labels[predicted_label_index]

    # Remove the temporarily saved image
    os.remove(image_path)

    return jsonify({'predicted_label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
