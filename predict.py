import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="chilli_model_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Healthy', 'Curl', 'Yellowish', 'Whitefly']

def predict_image(image_file):
    image = Image.open(image_file).resize((128, 128))
    img_array = np.array(image) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = class_names[np.argmax(output_data)]

    return predicted_class
