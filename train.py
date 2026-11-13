"""
=============================================================================
Safety PPE (Personal Protective Equipment) Detection Training Script
Framework: Ultralytics YOLOv8 (PyTorch)
Hardware: NVIDIA GeForce RTX 3050 (CUDA GPU)
Dataset: 6,629 images (Train: 5001, Val: 1328, Test: 300)
Classes: ['No_BreathingApparatus', 'No_Glove', 'No_Goggles', 'No_Harness', 
          'No_Helmet', 'No_Shoe', 'Person']
=============================================================================
"""

import os
from ultralytics import YOLO
import torch

def main():
    # 1. Print System & GPU Information
    print("=" * 60)
    print("  Smart Safety PPE Detection - Model Training Pipeline")
    print("=" * 60)
    print(f"PyTorch Version  : {torch.__version__}")
    print(f"CUDA Available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Active GPU Device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # 2. Path to dataset YAML configuration
    # data.yaml contains train, val, and class label mappings
    dataset_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "data.yaml")
    
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"Dataset config not found at: {dataset_yaml}")
        
    print(f"Using Dataset Configuration: {dataset_yaml}")

    # 3. Initialize YOLOv8 Model
    # We use YOLOv8n (Nano) - pretrained on COCO, optimized for real-time edge & desktop inference
    print("\n[INFO] Loading pretrained YOLOv8n backbone...")
    model = YOLO("yolov8n.pt")

    # 4. Train the Model
    # Batch size 16 is optimal for RTX 3050 4GB VRAM
    # imgsz=640 is standard high-resolution input for YOLOv8
    print("\n[INFO] Starting Fine-Tuning on Safety PPE Dataset...")
    results = model.train(
        data=dataset_yaml,
        epochs=15,             # 15 epochs gives strong convergence on 5,000 images
        imgsz=640,
        batch=16,              # Fits safely inside 4GB VRAM
        device=0,              # CUDA GPU:0
        workers=2,             # Multi-worker dataloading for Windows
        project="runs/detect", # Directory where weights and metric plots are saved
        name="ppe_yolov8",     # Run name
        exist_ok=True,         # Overwrite existing run if re-running
        pretrained=True,       # Transfer learning from COCO
        patience=5,            # Early stopping patience
        save=True,
        verbose=True
    )

    print("\n" + "=" * 60)
    print(" TRAINING COMPLETED SUCCESSFULLY!")
    print(f" Best Weights saved at: runs/detect/ppe_yolov8/weights/best.pt")
    print(f" Metric plots saved at: runs/detect/ppe_yolov8/")
    print("=" * 60)

if __name__ == "__main__":
    main()
