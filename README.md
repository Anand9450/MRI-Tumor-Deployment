---
title: NeuroScan AI
emoji: 🧠
colorFrom: indigo
colorTo: rose
sdk: gradio
sdk_version: 5.7.1
app_file: app.py
pinned: false
license: mit
---

# 🧠 NeuroScan AI (Brain Tumor Classification)

Advanced MRI analysis for early detection of Glioma, Meningioma, and Pituitary tumors. This project implements a deep learning model deployed as an interactive web application.

## 🚀 Live Application
[**Try the Demo Here**](https://huggingface.co/spaces/anand9450/MRI-Brain-Tumor-Classifier)

## Overview

The model uses a Convolutional Neural Network (CNN) built with **TensorFlow/Keras** to classify brain MRI scans into four categories:
*   **Glioma Tumor**
*   **Meningioma Tumor**
*   **Pituitary Tumor**
*   **No Tumor**

## How to Use

1.  Upload a clear MRI scan (T1-weighted contrast-enhanced images recommended).
2.  Click **Analyze Scan**.
3.  View the predicted diagnosis and confidence scores.

## How to Run Locally

1.  Clone this repository: `git clone https://github.com/Anand9450/MRI-Tumor-Deployment.git`
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the application:
    ```bash
    python app.py
    ```

## Disclaimer

**For Educational Purposes Only.** This tool is not a substitute for professional medical advice, diagnosis, or treatment.
