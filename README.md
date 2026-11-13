# 🛡️ Deep Learning-Based Industrial & Construction Safety PPE Detection System

An intelligent, real-time Deep Learning & Computer Vision surveillance system designed to detect Personal Protective Equipment (PPE) violations among industrial workers.

---

## 🎯 Project Overview & Objective
In construction sites and industrial manufacturing plants, workplace hazards cause thousands of avoidable injuries every year due to workers failing to wear mandatory safety gear. 
This project uses **YOLOv8 (Ultralytics on PyTorch)** to automatically detect workers and inspect compliance for:
- 👷 **Safety Helmets (Hard Hats)** (`No_Helmet`)
- 🥾 **Safety Shoes / Steel-Toe Boots** (`No_Shoe`)
- 🧤 **Protective Gloves** (`No_Glove`)
- 🥽 **Safety Goggles** (`No_Goggles`)
- 🦺 **Safety Harnesses** (`No_Harness`)
- 😷 **Breathing Respirators** (`No_BreathingApparatus`)

---

## 🗂️ Project Directory Structure
```text
C:\Users\sachi\Desktop\computer_vision_project\
│
├── dataset/                        # 6,629 image dataset (Roboflow Safety PPE)
│   ├── train/images & labels       # 5,001 training images
│   ├── valid/images & labels       # 1,328 validation images
│   ├── test/images & labels        # 300 test images
│   └── data.yaml                   # YOLO dataset configuration
│
├── train_model.ipynb               # 📓 Interactive Jupyter Notebook to train & visualize
├── train.py                        # Python standalone training script
├── app.py                          # 🚀 Streamlit Interactive Web Dashboard
├── run_app.bat                     # ⚡ 1-Click launcher for Streamlit Web Dashboard
├── check.ipynb                     # GPU verification test notebook
└── README.md                       # Complete Project Documentation & Viva Guide
```

---

## 🚀 How to Run the Project (Step-by-Step)

### Step 1: Open & Run the Training Notebook (`train_model.ipynb`)
1. Open this folder in **VS Code** or launch **Jupyter Notebook**.
2. Open `train_model.ipynb`.
3. In the top-right corner of the notebook, select the Python Kernel:
   - Choose: **`pytorch_env`** (`Python 3.10.x`)
4. Run the cells step-by-step:
   - **Cell 1:** Checks PyTorch & verifies RTX 3050 GPU is active.
   - **Cell 2:** Displays class labels and dataset count (6,629 images).
   - **Cell 3:** Previews raw dataset images.
   - **Cell 4:** Starts model training on GPU with live epoch progress bars.
   - **Cell 5:** Visualizes Training Loss curves & Confusion Matrix.
   - **Cell 6:** Tests prediction on a test image and prints compliance report.

> **Note:** Once trained, your model weights will be saved automatically to:
> `runs/detect/ppe_yolov8/weights/best.pt`

---

### Step 2: Launch the Web Dashboard (`app.py`)
You can launch the web dashboard in either of two ways:

#### Option A (1-Click Shortcut):
- Simply double-click **`run_app.bat`**.

#### Option B (Command Line):
1. Open terminal/PowerShell in `pytorch_env`:
   ```bash
   conda activate pytorch_env
   cd C:\Users\sachi\Desktop\computer_vision_project
   streamlit run app.py
   ```
2. Your browser will open at `http://localhost:8501`.

---

## 🌟 Web Dashboard Features
1. **Drag & Drop Image Inspection:**
   - Drop any site/worker photo.
   - Instant detection with red/green bounding boxes.
   - Safety compliance summary card (Compliant vs Non-Compliant).
   - Downloadable CSV audit report.
2. **Viva 1-Click Test Gallery:**
   - Pre-loaded sample images from `dataset/test/images` ready for 1-click evaluation without needing a webcam or taking live pictures.
3. **Viva Q&A & Performance Tab:**
   - Display of training loss curves and confusion matrix.
   - Built-in answers to standard viva questions.

---

## 🎓 College Viva Preparation (Top 5 Questions & Answers)

### Q1: Why did you choose YOLOv8 over previous YOLO versions (YOLOv5/v4)?
* **Answer:** YOLOv8 features an **anchor-free split head** architecture. Previous versions used predefined anchor boxes that required manual tuning and performed poorly on diverse object scales. YOLOv8 directly predicts object centers and bounding box offsets, resulting in faster convergence and higher mAP (Mean Average Precision).

### Q2: Why PyTorch instead of TensorFlow?
* **Answer:** Modern computer vision models (Ultralytics YOLOv8, Meta SAM, DETR) are natively written and maintained in PyTorch. PyTorch supports native YOLO text annotations directly (no TFRecord conversion required) and has active native CUDA GPU support on Windows.

### Q3: What loss functions does YOLOv8 use?
* **Answer:** YOLOv8 utilizes a multi-task loss function:
  1. **BCE (Binary Cross Entropy) Loss:** For class classification.
  2. **CIoU (Complete Intersection over Union) Loss:** For accurate bounding box spatial overlap, aspect ratio, and center distance.
  3. **DFL (Distribution Focal Loss):** For precise regression around object boundaries and handling partial occlusions.

### Q4: What is mAP (Mean Average Precision)?
* **Answer:** mAP measures detection accuracy across all classes. `mAP@50` evaluates accuracy when predicted bounding boxes overlap ground truth by at least 50% IoU. `mAP@50-95` averages mAP across thresholds from 0.50 to 0.95, representing overall model localization quality.

### Q5: What is Non-Maximum Suppression (NMS)?
* **Answer:** An object detector typically outputs hundreds of candidate bounding boxes per object. NMS keeps the box with the highest confidence and eliminates overlapping boxes whose IoU exceeds the set threshold (e.g. 0.45).
