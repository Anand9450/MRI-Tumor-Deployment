# 🧠 Brain Tumor Classification Web Application

This project implements a deep learning model for classifying brain tumors (Glioma, Meningioma, Pituitary) from MRI scans. The model is deployed as an interactive web application using Gradio.

## 🚀 Live Application

[Link to Hugging Face Space (To be added after deployment)]

## 📋 Table of Contents

- [Model Architecture](#model-architecture)
- [Classification Classes](#classification-classes)
- [How to Run Locally](#how-to-run-locally)
- [Deployment](#deployment)

---

## ✨ Model Architecture

The model uses a Convolutional Neural Network (CNN) built with **TensorFlow/Keras**. It was trained on the large MRI image dataset to classify images into four categories:

* **Glioma Tumor**
* **Meningioma Tumor**
* **Pituitary Tumor**
* **No Tumor** (Notumor)

The original model file (`tumor_detection_model.h5` at **122 MB**) is tracked using Git Large File Storage (Git LFS).

## 🗃️ Classification Classes

The model predicts one of the following four labels:

| Class Index | Label | Result |
| :---: | :--- | :--- |
| 0 | `glioma` | Tumor: Glioma |
| 1 | `meningioma` | Tumor: Meningioma |
| 2 | `notumor` | No Tumor Detected |
| 3 | `pituitary` | Tumor: Pituitary |

## 💻 How to Run Locally

You can run the web application directly on your local machine using the required dependencies.

### Prerequisites

1.  Clone this repository: `git clone https://github.com/Anand9450/MRI-Tumor-Deployment.git`
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Execution

Run the Gradio application script:

```bash
python app.py
