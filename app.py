import os
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr

# --- 1. Load Model and Define Constants ---
# TODO: Replace with your actual model path
MODEL_PATH = 'tumor_detection_model.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    # If the model fails to load, keep `model` as None and handle gracefully later
    model = None

# TODO: Define your image size used for training
# Use the same size you trained the model with (notebook used 128)
IMAGE_SIZE = 128

# TODO: Define your class list (make sure order matches model output)
# Example: If your classes are 'glioma', 'meningioma', 'notumor', 'pituitary'
unique_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
# If you don't have unique_labels, you might need to load it from your training setup
# or manually list the classes in the order they were processed.

# --- 2. Gradio-Compatible Prediction Function ---
def classify_brain_mri(input_img):
    """
    Predicts the tumor type from a PIL Image object and returns the
    image with the overlay and the result text.
    """
    try:
        # Ensure the model is loaded
        if model is None:
            return None, "Model not loaded. Check server logs for loading errors."

        # Resize and preprocess the image
        pil_img = input_img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(pil_img) / 255.0
        batch = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Prediction
        prediction = model.predict(batch)
        predicted_class_index = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))

        # Determine class and result
        class_list = unique_labels
        predicted_label = class_list[predicted_class_index]

        # Standardize result for display
        if predicted_label in ('notumor', 'no_tumor', 'no-tumor', 'no tumor'):
            result_text = 'No Tumor Detected'
        else:
            result_text = f"Tumor Type: {predicted_label.replace('-', ' ').title()}"

        # Create image with title overlay (drawn via Matplotlib), then convert to PIL image
        fig, ax = plt.subplots()
        ax.imshow(input_img)
        ax.axis('off')

        title_text = f'Prediction: {result_text}\nConfidence: {confidence*100:.2f}%'
        ax.set_title(title_text, fontsize=12, color='white', backgroundcolor='black')

        # Save the plot to an in-memory buffer and convert to a PIL Image for Gradio
        buf = io.BytesIO()
        plt.tight_layout(pad=0)
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        result_img = Image.open(buf).convert('RGB')
        plt.close(fig)

        return result_img, title_text

    except Exception as e:
        # In a web app, it's better to return an error message
        error_text = f"An error occurred during prediction: {e}"
        return None, error_text

# --- 3. Gradio Interface Setup ---

# Define the components for the UI
input_component = gr.Image(type="pil", label="Upload Brain MRI Image")
# Define the outputs: one for the image display, one for the text result
output_components = [
    gr.Image(type="pil", label="Processed Image with Prediction"),
    gr.Textbox(label="Classification Result")
]

# Create the Gradio Interface
gr.Interface(
    fn=classify_brain_mri,
    inputs=input_component,
    outputs=output_components,
    title="🧠 Brain Tumor Classification using MRI",
    description="Upload a T1-weighted contrast-enhanced MRI scan to classify it as **Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor**.",
    theme=gr.themes.Soft(), # A beautiful theme for a clean look
    live=True # Predicts as soon as the image is uploaded
).launch(share=True) # Set share=True to get a public link (temporarily)