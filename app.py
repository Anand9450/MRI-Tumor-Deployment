import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import gradio as gr 
from PIL import Image

# --- 1. Load Model and Define Constants ---
MODEL_PATH = 'tumor_detection_model.h5'
IMAGE_SIZE = 128
UNIQUE_LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- 2. Prediction Function ---
def classify_brain_mri(input_img):
    """
    Predicts the tumor type from a PIL Image object.
    Returns the top predicted label with confidence and a dictionary of all probabilities.
    """
    if model is None:
        return None, {"Error": 0.0}
    
    if input_img is None:
        return None, None

    try:
        # Resize and preprocess the image
        pil_img = input_img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(pil_img) / 255.0
        batch = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction_probs = model.predict(batch)[0]
        
        # Create a dictionary of labels and probabilities
        # Check if the model output shape matches the labels
        if len(prediction_probs) != len(UNIQUE_LABELS):
            # Fallback if model output size doesn't match expected labels
            # This is a safety check.
            labels = [f"Class {i}" for i in range(len(prediction_probs))]
            confidences = {labels[i]: float(prediction_probs[i]) for i in range(len(prediction_probs))}
        else:
            confidences = {UNIQUE_LABELS[i]: float(prediction_probs[i]) for i in range(len(UNIQUE_LABELS))}

        # Get the top prediction for the text output
        top_prediction_index = np.argmax(prediction_probs)
        top_label = list(confidences.keys())[top_prediction_index]
        top_confidence = prediction_probs[top_prediction_index]
        
        result_text = f"## Diagnosis: **{top_label}**\n### Confidence: **{top_confidence*100:.2f}%**"
        
        return result_text, confidences

    except Exception as e:
        return f"Error: {str(e)}", None

# --- 3. Custom CSS for a Premium UI ---
custom_css = """
.container {
    max-width: 1200px;
    margin: auto;
    padding-top: 20px;
}
.header {
    text-align: center;
    margin-bottom: 2rem;
}
.header h1 {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4F46E5 0%, #E11D48 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.header p {
    font-size: 1.2rem;
    color: #6B7280;
}
.gr-button-primary {
    background: linear-gradient(90deg, #4F46E5 0%, #6366F1 100%);
    border: none;
    color: white;
}
.footer {
    text-align: center;
    margin-top: 2rem;
    color: #9CA3AF;
    font-size: 0.875rem;
}
"""

# --- 4. Gradio Blocks Interface ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="rose"), css=custom_css, title="NeuroScan AI") as app:
    
    with gr.Column(elem_classes=["container"]):
        with gr.Column(elem_classes=["header"]):
            gr.Markdown("# 🧠 NeuroScan AI: Tumor Detection")
            gr.Markdown("Advanced MRI analysis for early detection of Glioma, Meningioma, and Pituitary tumors.")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Brain MRI", height=400)
                
                with gr.Row():
                    clear_btn = gr.ClearButton(components=[image_input], variant="secondary")
                    submit_btn = gr.Button("Analyze Scan", variant="primary", elem_classes=["gr-button-primary"])
                
                gr.Markdown("### Examples")
                examples = gr.Examples(
                    examples=[
                        ["examples/glioma.jpg"],
                        ["examples/meningioma.jpg"],
                        ["examples/pituitary.jpg"],
                        ["examples/notumor.jpg"]
                    ],
                    inputs=image_input
                )

            with gr.Column(scale=1):
                gr.Markdown("### Analysis Results")
                result_text = gr.Markdown("Please upload an image and click 'Analyze Scan'.")
                confidences_plot = gr.Label(label="Confidence Scores")
                
                with gr.Accordion("Understanding the Results", open=False):
                    gr.Markdown("""
                    - **Glioma**: A type of tumor that occurs in the brain and spinal cord.
                    - **Meningioma**: A tumor that arises from the meninges — the membranes that surround your brain.
                    - **Pituitary**: A tumor that forms in the pituitary gland.
                    - **No Tumor**: No abnormal mass detected in the scan.
                    
                    *Disclaimer: This tool is for educational purposes only and should not be used as a substitute for professional medical advice.*
                    """)

        gr.Markdown("© 2024 NeuroScan AI Project | Powered by TensorFlow & Gradio", elem_classes=["footer"])

    # Event handlers
    submit_btn.click(
        fn=classify_brain_mri,
        inputs=image_input,
        outputs=[result_text, confidences_plot]
    )

if __name__ == "__main__":
    app.launch(share=True)