import os
import argparse
import shutil
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 Waste Segregation Model Locally")
    parser.add_argument(
        "--model", 
        type=str, 
        default="n", 
        choices=["n", "s", "nano", "small"], 
        help="YOLO11 model version to train: 'n' (nano, default) or 's' (small)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    args = parser.parse_args()

    # Determine model filename based on choice
    model_choice = args.model.lower()
    if model_choice in ["n", "nano"]:
        model_name = "yolo11n.pt"
    else:
        model_name = "yolo11s.pt"

    print("=== Waste Segregation Local Training Pipeline ===")
    print(f"Selected model: {model_name}")

    # Set up paths
    data_yaml_path = r"C:\Users\repal\Downloads\Waste Object Detection.v6i.yolov11\data.yaml"
    if not os.path.exists(data_yaml_path):
        print(f"Error: Dataset data.yaml not found at path: {data_yaml_path}")
        print("Please ensure your dataset is extracted to that exact location.")
        return

    # Load model
    print(f"Loading pre-trained weights for {model_name}...")
    model = YOLO(model_name)

    # Automatically check for GPU acceleration (integrated Intel graphics will fallback to CPU)
    import torch
    if torch.cuda.is_available():
        device = 0
        print("CUDA GPU detected! Training will run on GPU (device 0).")
    else:
        device = "cpu"
        print("No CUDA GPU detected (Integrated Intel Graphics fallback). Training will run on CPU.")

    print(f"Starting training on dataset: {data_yaml_path}...")
    # Train the model
    # Note: imgsz=640, standard augmentations are enabled by default in YOLOv11 (e.g. mosaic, scale, fliplr).
    # Since we need to support both single-object and multi-object, we keep standard augmentations.
    # To handle single-object images well, we set rect=False (default) and standard scale/translate,
    # ensuring it generalizes well to large objects occupying most of the frame.
    results = model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        imgsz=640,
        device=device,
        project="runs/detect",
        name="train",
        exist_ok=True
    )

    print("\n=== Training Completed! ===")
    
    # Retrieve metrics from results
    # results.results_dict contains metrics keys:
    # metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
    metrics = results.results_dict
    precision = metrics.get("metrics/precision(B)", 0.0)
    recall = metrics.get("metrics/recall(B)", 0.0)
    mAP50 = metrics.get("metrics/mAP50(B)", 0.0)
    
    # Calculate F1 Score
    f1_score = 0.0
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"Training Metrics:")
    print(f" - Precision: {precision:.4f}")
    print(f" - Recall: {recall:.4f}")
    print(f" - F1 Score: {f1_score:.4f}")
    print(f" - mAP50: {mAP50:.4f}")

    # Paths to copy weights
    best_weights_source = os.path.join("runs", "detect", "train", "weights", "best.pt")
    best_weights_dest = "best.pt"

    if os.path.exists(best_weights_source):
        shutil.copy(best_weights_source, best_weights_dest)
        print(f"Success: Copied model weights to root path: {os.path.abspath(best_weights_dest)}")
    else:
        print(f"Warning: Could not find trained weights at {best_weights_source}")
        return

    # Export to ONNX format
    print("Exporting trained model to ONNX format...")
    try:
        # Load the newly saved best.pt to ensure we export the trained weights
        trained_model = YOLO(best_weights_dest)
        onnx_path = trained_model.export(format="onnx", imgsz=640)
        print(f"Success: Model exported to ONNX format at: {onnx_path}")
    except Exception as e:
        print(f"Error during ONNX export: {e}")

if __name__ == "__main__":
    main()
