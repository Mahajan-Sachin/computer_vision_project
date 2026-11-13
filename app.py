import os
import glob
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Deep Learning Safety PPE Compliance Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling (Professional Clean Theme)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-card-danger {
        background-color: #FEE2E2;
        border-left: 6px solid #DC2626;
        padding: 16px;
        border-radius: 8px;
        color: #991B1B;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .status-card-success {
        background-color: #DCFCE7;
        border-left: 6px solid #16A34A;
        padding: 16px;
        border-radius: 8px;
        color: #166534;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }
    .badge-violation {
        background-color: #FCA5A5;
        color: #7F1D1D;
    }
    .badge-ok {
        background-color: #86EFAC;
        color: #14532D;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Paths and Defaults
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "ppe_yolov8", "weights", "best.pt")
FALLBACK_MODEL = "yolov8n.pt"
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "test", "images")
TRAINING_METRICS_DIR = os.path.join(BASE_DIR, "runs", "detect", "ppe_yolov8")

CLASS_NAMES = [
    'No_BreathingApparatus', 
    'No_Glove', 
    'No_Goggles', 
    'No_Harness', 
    'No_Helmet', 
    'No_Shoe', 
    'Person'
]

VIOLATION_CLASSES = {
    'No_Helmet': '⚠️ Missing Safety Helmet (Hard Hat)',
    'No_Shoe': '⚠️ Missing Safety Footwear/Boots',
    'No_Glove': '⚠️ Missing Protective Gloves',
    'No_Goggles': '⚠️ Missing Eye Protection / Goggles',
    'No_Harness': '⚠️ Missing Safety Fall Harness',
    'No_BreathingApparatus': '⚠️ Missing Breathing Mask / Respirator'
}

# -----------------------------------------------------------------------------
# Load Model Function with Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model(weights_path):
    if os.path.exists(weights_path):
        return YOLO(weights_path), f"Fine-tuned PPE Model (`{weights_path}`)"
    
    # Auto-find any best.pt in runs directory
    found = glob.glob(os.path.join(BASE_DIR, "runs", "**", "best.pt"), recursive=True)
    if found:
        return YOLO(found[0]), f"Fine-tuned PPE Model (`{found[0]}`)"
        
    if os.path.exists(FALLBACK_MODEL):
        return YOLO(FALLBACK_MODEL), "Pretrained YOLOv8n (Baseline)"
    else:
        return YOLO("yolov8n.pt"), "Downloaded Pretrained YOLOv8n"

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/safety-hat.png", width=64)
    st.title("System Controls")
    
    # Model Loading
    model, model_info = load_yolo_model(MODEL_WEIGHTS)
    st.info(f"**Active Model:**\n{model_info}")
    
    st.markdown("---")
    st.subheader("Inference Settings")
    conf_threshold = st.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05,
                               help="Confidence threshold for bounding box detection. Recommended: 0.25")
    iou_threshold = st.slider("IoU (NMS) Threshold", min_value=0.10, max_value=0.90, value=0.45, step=0.05,
                             help="Threshold for overlapping bounding box suppression.")
    
    st.markdown("---")
    st.markdown("### Project Info")
    st.caption("**College Project:** Real-Time PPE & Workplace Safety Violation Detector")
    st.caption("**Framework:** PyTorch & Ultralytics YOLOv8")
    st.caption("**Hardware:** NVIDIA RTX 3050 GPU")

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🛡️ Industrial & Construction Safety PPE Violation Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Computer Vision Compliance System for Worker Safety Surveillance</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tabs Interface
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🖼️ Image Inspection & Live Audit", "⚡ Viva Presentation Mode (1-Click Test)", "📊 Model Metrics & Viva Q&A"])

# =============================================================================
# TAB 1: Image Upload & Live Audit
# =============================================================================
with tab1:
    col_up, col_info = st.columns([2, 1])
    with col_up:
        input_mode = st.radio("Choose Input Method:", ["📁 Upload / Drop Image from PC", "📷 Live Webcam Capture"], horizontal=True)
        if input_mode == "📁 Upload / Drop Image from PC":
            uploaded_file = st.file_uploader("Drag & drop or browse worker/site image (JPG, PNG, JPEG):", type=["jpg", "jpeg", "png"])
        else:
            uploaded_file = st.camera_input("Capture picture with your webcam")
            
    with col_info:
        st.markdown("""
        **🔍 Inspection Rules:**
        - Checks for mandatory PPE: Helmet, Boots, Gloves, Goggles, Harness.
        - Automatically detects violation bounding boxes.
        - Generates instant site safety compliance audit report.
        """)
        
    image_to_process = None
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file).convert("RGB")
        
    if image_to_process is not None:
        st.markdown("---")
        
        # Run YOLO Inference
        with st.spinner("Analyzing worker safety compliance with YOLOv8..."):
            results = model.predict(source=image_to_process, conf=conf_threshold, iou=iou_threshold)
            res = results[0]
            
            # Annotated image as PIL Image
            annotated_img_bgr = res.plot()
            # Convert BGR to RGB
            annotated_img_rgb = annotated_img_bgr[:, :, ::-1]
            annotated_pil = Image.fromarray(annotated_img_rgb)
            
            # Parse detections
            detections = []
            boxes = res.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = res.names[cls_id]
                conf = float(box.conf[0].item())
                detections.append({
                    "Class": cls_name,
                    "Confidence": f"{conf:.2%}",
                    "Type": "Violation" if cls_name.startswith("No_") else "Safe/Entity"
                })
                
        # Layout: Side by side images
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Original Input Image")
            st.image(image_to_process, use_container_width=True)
        with col2:
            st.subheader("🎯 Model Detection Output")
            st.image(annotated_pil, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Safety Compliance Audit Summary")
        
        # Calculate violations
        violations = [d for d in detections if d["Class"].startswith("No_")]
        persons = [d for d in detections if d["Class"] == "Person"]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("👷 Persons Detected", len(persons))
        m2.metric("⚠️ Safety Violations Found", len(violations))
        
        if len(violations) == 0:
            m3.metric("🛡️ Site Status", "100% COMPLIANT", delta="Safe")
            st.markdown("""
            <div class="status-card-success">
                ✅ <b>ALL COMPLIANCE STANDARDS MET:</b> No PPE safety violations detected for the monitored worker(s).
            </div>
            """, unsafe_allow_html=True)
        else:
            m3.metric("🚨 Site Status", "NON-COMPLIANT", delta="-Hazard", delta_color="inverse")
            st.markdown(f"""
            <div class="status-card-danger">
                🚨 <b>SAFETY VIOLATION ALERT:</b> {len(violations)} safety breach(es) detected! Immediate supervisor intervention required.
            </div>
            """, unsafe_allow_html=True)

        # Detailed breakdown table
        if detections:
            df = pd.DataFrame(detections)
            st.markdown("#### Detected Objects Breakdown")
            st.dataframe(df, use_container_width=True)
            
            # CSV Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Safety Audit Report (CSV)", data=csv, file_name="safety_audit_report.csv", mime="text/csv")

# =============================================================================
# TAB 2: Viva Presentation Mode (1-Click Test Samples)
# =============================================================================
with tab2:
    st.subheader("⚡ 1-Click Test Gallery (Pre-loaded from Dataset)")
    st.markdown("Use this tab during your viva presentation to test real dataset images instantly without needing to search or take photos live!")

    # Check if test images exist
    sample_images = []
    if os.path.exists(TEST_IMAGES_DIR):
        all_test_imgs = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg")) + glob.glob(os.path.join(TEST_IMAGES_DIR, "*.png"))
        sample_images = all_test_imgs[:12] # First 12 test samples

    if not sample_images:
        st.warning(f"No test images found in `{TEST_IMAGES_DIR}`. Please verify dataset path.")
    else:
        st.write(f"Showing **{len(sample_images)} curated test samples** from the dataset:")
        
        # Grid of sample images to pick
        cols = st.columns(4)
        selected_sample_idx = None
        
        for idx, img_path in enumerate(sample_images[:8]):
            with cols[idx % 4]:
                thumb = Image.open(img_path)
                st.image(thumb, caption=f"Sample {idx+1}", use_container_width=True)
                if st.button(f"🔍 Inspect Sample {idx+1}", key=f"btn_sample_{idx}"):
                    st.session_state["selected_sample_path"] = img_path

        # If a sample is selected, run inference
        if "selected_sample_path" in st.session_state and os.path.exists(st.session_state["selected_sample_path"]):
            selected_path = st.session_state["selected_sample_path"]
            st.markdown("---")
            st.markdown(f"### Running Inspection on: `{os.path.basename(selected_path)}`")
            
            sample_img = Image.open(selected_path).convert("RGB")
            with st.spinner("Analyzing..."):
                results = model.predict(source=sample_img, conf=conf_threshold, iou=iou_threshold)
                res = results[0]
                annotated_img_rgb = res.plot()[:, :, ::-1]
                
            c1, c2 = st.columns(2)
            with c1:
                st.image(sample_img, caption="Input Sample", use_container_width=True)
            with c2:
                st.image(annotated_img_rgb, caption="Detection Result", use_container_width=True)
                
            # Parse sample detections
            sample_detections = []
            for b in res.boxes:
                c_id = int(b.cls[0].item())
                c_name = res.names[c_id]
                c_conf = float(b.conf[0].item())
                sample_detections.append({
                    "Class": c_name,
                    "Confidence": f"{c_conf:.2%}",
                    "Type": "Violation" if c_name.startswith("No_") else "Safe/Entity"
                })
                
            sample_violations = [d for d in sample_detections if d["Class"].startswith("No_")]
            sample_persons = [d for d in sample_detections if d["Class"] == "Person"]
            
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("👷 Persons Detected", len(sample_persons))
            m2.metric("⚠️ Safety Violations Found", len(sample_violations))
            
            if len(sample_violations) == 0:
                m3.metric("🛡️ Site Status", "100% COMPLIANT", delta="Safe")
                st.markdown("""
                <div class="status-card-success">
                    ✅ <b>ALL COMPLIANCE STANDARDS MET:</b> No PPE safety violations detected.
                </div>
                """, unsafe_allow_html=True)
            else:
                m3.metric("🚨 Site Status", "NON-COMPLIANT", delta="-Hazard", delta_color="inverse")
                st.markdown(f"""
                <div class="status-card-danger">
                    🚨 <b>SAFETY VIOLATION ALERT:</b> {len(sample_violations)} safety breach(es) detected!
                </div>
                """, unsafe_allow_html=True)
                
            if sample_detections:
                sample_df = pd.DataFrame(sample_detections)
                st.dataframe(sample_df, use_container_width=True)

# =============================================================================
# TAB 3: Model Metrics & Viva Q&A Guide
# =============================================================================
with tab3:
    st.subheader("📊 Model Performance & Training Curves")
    
    # Check for saved training metric plots
    results_png = os.path.join(TRAINING_METRICS_DIR, "results.png")
    confusion_png = os.path.join(TRAINING_METRICS_DIR, "confusion_matrix.png")
    val_labels_png = os.path.join(TRAINING_METRICS_DIR, "val_batch0_labels.jpg")
    
    if os.path.exists(results_png):
        st.markdown("#### 1. Training & Validation Loss / mAP Curves")
        st.image(results_png, caption="YOLOv8 Training Metrics (Losses & Mean Average Precision)", use_container_width=True)
    else:
        st.info("ℹ️ Training curves will appear here once you run `python train.py` to train the model.")

    if os.path.exists(confusion_png):
        st.markdown("#### 2. Confusion Matrix")
        st.image(confusion_png, caption="Class-wise Confusion Matrix", use_container_width=True)

    st.markdown("---")
    st.subheader("🎓 College Viva Q&A Cheat Sheet")
    st.caption("Common questions examiners ask for this exact project:")

    with st.expander("Q1: What is YOLOv8 and why is it called an 'Anchor-Free' model?"):
        st.markdown("""
        **Answer:**
        - YOLO (You Only Look Once) is a single-stage object detection architecture.
        - Older models (YOLOv3, YOLOv4, YOLOv5) used predefined **Anchor Boxes** (fixed bounding box shapes that required manual tuning).
        - **YOLOv8 is Anchor-Free**: It directly predicts the center of an object and bounding box offsets. This reduces hyperparameter tuning, handles diverse aspect ratios better, and improves inference speed.
        """)

    with st.expander("Q2: Why did you choose PyTorch over TensorFlow?"):
        st.markdown("""
        **Answer:**
        1. **Industry Standard for CV:** Modern state-of-the-art vision models (Ultralytics YOLOv8, Meta's Segment Anything SAM) are developed natively in PyTorch.
        2. **Native YOLO Format:** PyTorch natively reads YOLO annotation text files without needing conversion to bulky TFRecords.
        3. **Native Windows GPU Acceleration:** PyTorch 2.5 with CUDA runs natively on modern Windows GPUs without requiring WSL2 Linux.
        """)

    with st.expander("Q3: What loss functions does YOLOv8 use?"):
        st.markdown("""
        **Answer:**
        YOLOv8 uses a combination of three loss functions:
        1. **BCE Loss (Binary Cross-Entropy):** For multi-label class classification.
        2. **CIoU Loss (Complete Intersection over Union):** For bounding box localization (takes into account overlap area, aspect ratio, and center distance).
        3. **DFL (Distribution Focal Loss):** Improves bounding box regression near edges and handles occluded objects.
        """)

    with st.expander("Q4: What is mAP (Mean Average Precision)?"):
        st.markdown("""
        **Answer:**
        - **mAP@50:** Mean Average Precision at an IoU (Intersection over Union) threshold of 0.50. It measures the percentage of correct detections where bounding boxes overlap with ground truth by at least 50%.
        - **mAP@50-95:** Average mAP across IoU thresholds from 0.50 to 0.95 with steps of 0.05, which is the benchmark metric for COCO evaluation.
        """)

    with st.expander("Q5: What is Non-Maximum Suppression (NMS)?"):
        st.markdown("""
        **Answer:**
        When an object detector runs, it often predicts multiple overlapping bounding boxes for a single person/object. **NMS** filters these out:
        1. Selects the box with the highest confidence score.
        2. Calculates the IoU of that box with all other overlapping boxes.
        3. If IoU > threshold (e.g., 0.45), the lower-confidence duplicate boxes are discarded.
        """)
