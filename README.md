# 🧠 Explainable AI (XAI) Platform for Robust Medical Image Analysis

An interactive, end-to-end deep learning framework designed to classify brain tumors from MRI scans while providing visual interpretability using **Grad-CAM (Gradient-weighted Class Activation Mapping)**. This platform bridges the gap between deep learning performance and clinical trust by transforming "black-box" neural predictions into explainable, heat-mapped diagnostic proofs.

---

## 🚀 Key Features

* **Automated Multi-Class Diagnosis:** Classifies brain MRI scans into four distinct pathological categories: *Glioma*, *Meningioma*, *Pituitary Tumor*, and *No Tumor (Healthy)*.
* **Explainable AI Engine:** Implements custom forward and backward PyTorch hooks to extract class-activation feature maps from the model's deepest convolutional block.
* **Interactive Clinician Dashboard:** A responsive, lightweight web application built with **Streamlit** that enables seamless medical scan uploading, real-time inference, and heat-map overlay rendering.
* **Modular Software Architecture:** Built following strict production standards with clear separation of concerns across datasets, model definitions, explainer tools, and execution runtimes.

---

## 🛠️ Tech Stack & Libraries

* **Core Framework:** `Python 3.10+`
* **Deep Learning & Computer Vision:** `PyTorch`, `Torchvision`, `OpenCV (opencv-python)`
* **Interactive Web UI:** `Streamlit`
* **Data Processing & Arrays:** `NumPy`, `Pillow (PIL)`

---

## 📂 Project Architecture

```text
Medical_XAI_Project/
│
├── data/
│   └── raw/
│       ├── Testing/            # Evaluation subdirectories (glioma, meningioma, etc.)
│       └── Training/           # Optimization subdirectories (glioma, meningioma, etc.)
│
├── models/
│   └── best_brain_tumor_model.pth  # Serialized optimal pre-trained weights
│
├── src/
│   ├── explainers/
│   │   └── gradcam.py          # Core Grad-CAM hook logic & math engine
│   ├── models/
│   │   └── model.py            # ResNet-18 Transfer Learning head modification
│   └── utils/
│       └── dataset.py          # PyTorch custom dataset image parser & pipeline
│
├── app.py                      # Interactive Streamlit Web Interface Dashboard
├── generate_xai.py             # Offline verification and image generation script
└── train.py                    # Optimization loop training & validation engine



