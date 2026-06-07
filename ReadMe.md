# Skin Lesion Classification CNN

## Project Log & Documentation

**Project Objective:** Build, evaluate, and deploy a custom Convolutional Neural Network to classify skin lesion images into Binary categories (Benign vs. Malignant) using the Streamlit framework.

**Hardware Constraints:**

- GPU: NVIDIA Quadro P620 (4 GB VRAM)
- Memory Growth: Enabled
- Max Image Resolution: 128x128
- Max Batch Size: 32

---

## Quick Start & Setup

To run the Medical Lesion Analyzer locally, follow these steps:

### 1. Environment Setup
Create a Python virtual environment and install the required dependencies:
```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install tensorflow==2.15.0 streamlit numpy Pillow opencv-python pandas scikit-learn matplotlib seaborn
```

### 2. Run the Application
Launch the Single Page Application (SPA) Streamlit UI:
```bash
streamlit run app.py
```
This will automatically open the clinical interface in your default web browser at `http://localhost:8501`.

---

### Execution Logs

**[2026-06-06] Phase 1: Planning & Setup**

- Skills Loaded: Computer Vision Expert, Python Pro, Data Scientist, ML Engineer.
- Project architecture defined (src/config.py, src/data_preprocessing.py, src/model.py, src/train.py, src/evaluate.py, app.py).
- Binary mapping configured:
  - Class 0 (Benign): NV, BKL, DF, VASC
  - Class 1 (Malignant): MEL, BCC, AKIEC
- Awaiting user approval of implementation plan to proceed with code generation.

**[2026-06-06] Phase 2: Data Preprocessing & Augmentation**

- Strict Guardrails Enforced:
  - Using `binary_crossentropy` with Class Weights (Focal Loss removed).
  - Image Augmentation applied exclusively to the training dataset.
  - `masks` directory strictly ignored.
  - Images streamed using `tf.data.Dataset` in batches of 32 at 128x128.
- Preprocessing Output:
  - Total valid images: 10015
  - Overall Distribution -> Benign (0): 8061, Malignant (1): 1954
  - Data splits -> Train: 7010 | Val: 1502 | Test: 1503
  - Train Distribution -> Benign (0): 5642, Malignant (1): 1368
- **Computed Class Weights (Manual Verification Required):**
  - Class 0 (Benign): `0.6212336051045728`
  - Class 1 (Malignant): `2.5621345029239766`
- Saved datasets and class weights to `data/splits/`.
- Paused execution to await manual approval of the computed Class Weights before starting Phase 3.

**[2026-06-07] Phase 2 on Linux**
--- Phase 2: Data Preprocessing ---

- Total valid images: 10015
- Overall Distribution -> Benign (0): 8061, Malignant (1): 1954
- Train split: 7010 | Val split: 1502 | Test split: 1503
- Computing class weights on training set ONLY...
- Train Distribution -> Benign (0): 5642, Malignant (1): 1368
- Computed Class Weights: {"0": 0.6212336051045728, "1": 2.5621345029239766}
- Data preprocessing complete and splits saved.

**[2026-06-06] Phase 3: CNN Architecture**

- Constructed a lightweight, custom Sequential CNN strictly tailored to the 4 GB VRAM limit. Pre-trained architectures (e.g., ResNet, VGG16) were ignored.
- **Architecture Blueprint:**
  - Input Layer: `(128, 128, 3)`
  - Conv Block 1: `Conv2D (32, 3x3, ReLU) -> MaxPooling2D (2x2)`
  - Conv Block 2: `Conv2D (64, 3x3, ReLU) -> MaxPooling2D (2x2)`
  - Conv Block 3: `Conv2D (128, 3x3, ReLU) -> MaxPooling2D (2x2)`
  - Flattening: `Flatten()`
  - Dense Block: `Dense (128, ReLU) -> Dropout (0.5)`
  - Output Layer: `Dense (1, Sigmoid)`
- **Total Trainable Parameters:** `3,304,769` (12.61 MB)
- Model instance saved successfully to `models/uncompiled_model.keras`.
- Paused execution to await manual approval before initiating Phase 4 (Training & Tuning).

**[2026-06-07] Phase 3 on Linux**

- 2026-06-07 10:42:42.879893: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1929] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 2865 MB memory:  -> device: 0, name: Quadro P620, pci bus id: 0000:01:00.0, compute capability: 6.1
Model: "SkinLesion_CNN"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d_1 (Conv2D)           (None, 126, 126, 32)      896

 max_pooling2d_1 (MaxPoolin  (None, 63, 63, 32)        0
 g2D)

 conv2d_2 (Conv2D)           (None, 61, 61, 64)        18496

 max_pooling2d_2 (MaxPoolin  (None, 30, 30, 64)        0
 g2D)

 conv2d_3 (Conv2D)           (None, 28, 28, 128)       73856

 max_pooling2d_3 (MaxPoolin  (None, 14, 14, 128)       0
 g2D)

 flatten (Flatten)           (None, 25088)             0

 dense_1 (Dense)             (None, 128)               3211392

 dropout (Dropout)           (None, 128)               0

 output (Dense)              (None, 1)                 129

=================================================================
Total params: 3304769 (12.61 MB)
Trainable params: 3304769 (12.61 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
Uncompiled model saved successfully to /mnt/d/Code/Projects/datacrumbs/datacrumbs_capstone_project_6/models/uncompiled_model.keras

**[2026-06-07] Phase 4: Training & Tuning (Hardware Accelerated)**

**Rigorous MLOps Interventions to Secure Hardware Acceleration:**
1. **WSL2 Migration & Python Lock:** The default Ubuntu 24.04 (Python 3.12) caused dependency failures. The System Architect manually injected the `deadsnakes` PPA, forced a `Python 3.10` environment, and locked TensorFlow to `==2.15.0`.
2. **NVIDIA Index Routing:** `tensorrt-libs` failed on the public PyPI. Bypassed this by appending `--extra-index-url https://pypi.nvidia.com` to the pip install command.
3. **Keras Serialization Fix:** Loading the Windows-generated `uncompiled_model.keras` in Linux caused a Keras 3 -> Keras 2 version crash. Resolved by manually re-executing `python3 -m src.model` in WSL2 to regenerate the architecture natively.
4. **Cross-OS Pathing Fix:** The Windows `D:\...` paths in the CSV splits caused a `NOT_FOUND` Graph Execution Error in Linux. Resolved by manually re-executing `python3 -m src.data_preprocessing` to regenerate the paths to the Linux `/mnt/d/...` standard.

**Phase 4 Final Verified Metrics:**
The network successfully processed the full 7,010 un-bottlenecked dataset per epoch on the NVIDIA Quadro P620 natively within WSL2.
- **Epoch 1:** `loss=0.6654`, `acc=0.5381`, `val_loss=0.5158`, `val_acc=0.5639`
- **Final Epoch:** `loss=0.3876`, `acc=0.7857`, `val_loss=0.4271`, `val_acc=0.7690`

Artifacts generated: `trained_lesion_classifier.keras` and `training_history.json`.

**[2026-06-07] Phase 5: Evaluation**
- Generated `src/evaluate.py` to run inference natively on the test set.
- A bypass flag (`compile=False`) was deployed to correctly deserialize the Keras 2 (`2.15.0`) model weights back into the Keras 3 host environment.
- **Visual Plots Generated:** `learning_curves` and `confusion_matrix` saved to `results/`.
- **Verified Test Metrics (Malignant Class):**
  - **Accuracy:** `0.7851`
  - **Precision:** `0.4702`
  - **Recall:** `0.8089`
  - **F1-Score:** `0.5947`
  - **Confusion Matrix:** `TN: 943 | FP: 267 | FN: 56 | TP: 237`
- **Analytical Summary:** The computed Class Weights highly prioritized minimizing False Negatives (FN) for the Malignant class. This successfully manifested in an excellent **Recall of 80.89%**—catching the vast majority of true malignant cases (237 out of 293). This high sensitivity comes at the cost of Precision (47%), resulting in 267 False Positives, which is the mathematically expected and medically preferred trade-off in early-stage lesion diagnostics (preferring a false alarm over a missed cancer).

**[2026-06-07] Phase 6: Streamlit Deployment**
- **UI Architecture:** Built a multi-page Streamlit application capturing the three core UI elements specified via the Google Stitch MCP ("Medical Lesion Analyzer" project):
  - `app.py`: Image Upload Screen (with drag-and-drop support).
  - `pages/1_Prediction_Result.py`: Prediction Result Screen (displaying binary mapping).
  - `pages/2_Confidence_Score.py`: Confidence Score Display (bar chart analytics).
- **Clinical Precision CSS Styling:** Integrated the custom Stitch CSS design system directly into the Streamlit rendering engine.
- **Cross-OS Engine Compatibility:** 
  - Implemented `pathlib.Path` to guarantee platform-agnostic file loading regardless of whether the Streamlit app is launched from the Windows Host or the WSL2 Linux environment.
  - Used `compile=False` when calling `load_model` to safely deserialize the WSL2-trained Keras 2 model inside native Windows Keras 3 environments.
- **Preprocessing Pipeline:** Injected identical mathematical scaling constraints into the Streamlit app before inference (Resize 128x128 -> Scale /255.0 -> Expand Dims).
- **UI Architectural Override (SPA):** Scrapped the Streamlit defaults and converted the app into a Single Page Application (SPA). Integrated the raw HTML/CSS from the "Clinical Precision" design system (Google Stitch) to generate dynamic, native frontend components (Result Card and Confidence Meter) directly bound to the ML outputs.

**[2026-06-07] Project Closure: Deployment**
- Project is officially complete and deployed to the live repository. The final architecture bridges a robust MLOps pipeline (WSL2/CUDA native training on NVIDIA Quadro P620) with a premium, clinical-grade Streamlit frontend interface.