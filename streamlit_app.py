import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import streamlit as st

# Load TFLite model
interpreter = tflite.Interpreter(model_path="chilli_model_quantized.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(image):
    img = image.resize((224, 224))
    data = np.expand_dims(np.array(img), axis=0).astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output)

# Streamlit UI
st.title("Chilli Disease Detector")
uploaded = st.file_uploader("Upload a chilli leaf image", type=["jpg", "png"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    if st.button("Predict"):
        label = predict(img)
        st.success(f"Predicted Class: {label}")
