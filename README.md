# 🧠 Multi-Modal Clinical Decision Support System (CDSS) & Explainable AI (XAI) Platform

[![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)](https://streamlit.io/)

An enterprise-grade, multi-modal Computer Vision and Data Fusion framework designed to classify brain tumors from MRI scans while providing multi-perspective visual interpretability, automated geometric localization, compliance privacy scrubbing, patient risk prioritization, and automated text report generation.

---

## 🚀 Advanced Platform Architecture

* **Dual-Engine XAI Suite:** Combines **Regional Coarse Localization (Grad-CAM)** and **Microscopic Pixel Attribution (Vanilla Saliency)** to provide radiologists with comprehensive structural evidence.
* **Multi-Modal Risk Fusion Engine:** Intelligently fuses deep learning image probabilities with patient demographic profiles (Age, Sex) and active clinical symptom vectors to compute a unified **Patient Priority Index**.
* **Automated CV Localization Box:** Leverages OpenCV thresholding and contour-tracing algorithms to calculate bounding geometry parameters directly around identified tumor masses.
* **Automated AI Clinical Report Engine:** Dynamically compiles vision-model parameters, classification logits, and fused risk indexes into a structured, downloadable text report (.TXT) for electronic health record integration.
* **Security & Ingestion Gateway:** Features a toggleable HIPAA/GDPR compliance layer that strips identifying file headers and metadata traces upon system ingestion, ensuring strict data privacy.
* **Post-Training Static Quantization:** Slashes the model memory footprint by **74.54%** (compressing the weights from **42.72 MB down to 10.87 MB**) using int8 quantization for efficient local CPU terminal deployments.

---

## ⚙️ Core Methodology

### 1. Data Engineering & Security
* **Privacy Scrubbing:** Recreates the raw image matrix upon ingestion, breaking any attachment to tracking strings or file origins.
* **Spatial Standardizations:** Dynamically reshapes input canvases to a uniform $224 \times 224 \times 3$ grid.
* **Distribution Shift Controls:** Normalizes incoming floating-point matrices using ImageNet baseline statistics ($\mu, \sigma$) to guarantee computational stability.

### 2. Multi-Modal Fusion Math
The unified **Patient Priority Index** ($P$) dynamically blends the vision network's target pathology confidence score ($C_{\text{AI}}$) with a structured clinical demographic hazard score ($H_{\text{Clinical}}$) using an operational ratio layout:

$$P = 0.7 \times C_{\text{AI}} + 0.3 \times H_{\text{Clinical}}$$

Where $H_{\text{Clinical}}$ is calculated programmatically based on a categorical symptom check matrix (e.g., severe headaches, seizures, visual disruptions) combined with age-adjusted vulnerability factors.

### 3. Interpretability Pipelines
* **Grad-CAM:** Captures spatial activation maps ($A^k$) and backpropagated gradients ($\frac{\partial Y^c}{\partial A^k}$) from `layer4` to calculate regional importance weights ($\alpha_k^c$), passing them through a **ReLU** filter to extract positive feature regions:
  $$L_{\text{Grad-CAM}}^c = \max\left(0, \sum_{k} \alpha_k^c A^k\right)$$
* **Vanilla Saliency:** Computes the direct analytical derivative of the highest target output class score relative to the raw input pixels, rendering high-contrast maps of delicate tissue edge distortions.

---

## 📈 System Performance Metrics

* **Diagnostic Performance:** Achieved **91.66% Training Accuracy** in the initial fine-tuning pass, stabilizing at **98.00% Validation Accuracy**.
* **Loss Optimization:** Validation cross-entropy loss cleanly minimized to a highly stable score of **0.0676**.
* **Edge Optimization:** Successfully compiled a **3.9x model compression ratio** via post-training static quantization, slashing space requirements from **42.72 MB down to 10.87 MB** to ensure rapid CPU performance.

---

## 📂 Repository Structure

```text
Medical_XAI_Project/
│
├── data/
│   └── raw/
│       ├── Testing/            # Evaluation splits (glioma, no_tumor, etc.)
│       └── Training/           # Optimization splits (glioma, no_tumor, etc.)
│
├── models/
│   ├── best_brain_tumor_model.pth       # Original FP32 Weights (42.72 MB)
│   └── quantized_brain_tumor_model.pth  # Compressed INT8 Edge Weights (10.87 MB)
│
├── src/
│   ├── explainers/
│   │   └── gradcam.py          # Dual XAI Engines (Grad-CAM & Vanilla Saliency)
│   ├── models/
│   │   └── model.py            # Modified ResNet-18 Architecture Head
│   └── utils/
│       └── dataset.py          # Custom PyTorch Dataset Image Parser
│
├── app.py                      # Multi-Tab Streamlit Production Workspace Portal
├── generate_xai.py             # Offline diagnostic verification script
├── quantize.py                 # Post-Training Static Quantization runtime
└── train.py                    # Multi-epoch training loop engine

💻 Local Installation & Setup
1. Initialize Runtime Environment
PowerShell
git clone [https://github.com/ArshSidhu540/Medical_XAI_Project.git](https://github.com/ArshSidhu540/Medical_XAI_Project.git)
cd Medical_XAI_Project
python -m venv .venv
.venv\Scripts\Activate.ps1
2. Ingest Dependencies
PowerShell
pip install torch torchvision opencv-python streamlit numpy Pillow scikit-learn plotly matplotlib
3. Pipeline Run Sequence
PowerShell
# 1. Train and fit the baseline transfer learning model weights
python train.py

# 2. Compress the serialized weights down to compact 8-bit architecture 
python quantize.py

# 3. Spin up the multi-modal production clinical dashboard local server
streamlit run app.py
The client workspace portal will automatically mount and open within your web browser at http://localhost:8501.

Developed as a Final Year Major Project in Computer Science & Engineering.


---

### Step 2: Push Your Code to GitHub

Open your terminal in VS Code and run these quick commands to send all your brilliant advancements directly up to your profile:

```powershell
# Stage the modified files
git add .

# Commit your final engineering additions
git commit -m "Feature: Integrated multi-modal data fusion, privacy gateway, and clinical reporting"

# Push to your remote repository
git push
You are officially ready! 🎓
You have taken this project through an exceptional evolution:

Phase 1 & 2: Handled custom data engineering, augmentation, and normalization.

Phase 3: Engineered a transfer learning head on a deep residual network.

Phase 4: Built custom forward/backward tensor hooks for Explainable AI.

Phase 5: Designed a complete, interactive, multi-tab analytics UI dashboard.

Advancements: Implemented computer vision localization, post-training parameters quantization, data privacy barriers, tabular metadata fusion, and dynamic text report creation.