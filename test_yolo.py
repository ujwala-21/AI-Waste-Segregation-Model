import os
from ultralytics import YOLO

# Load the local YOLO model
model_path = r"runs\detect\runs\detect\train\weights\best.pt"

print(f"Loading local YOLO model from: {model_path}")
model = YOLO(model_path)

# Run inference on local image 'Television.png'
image_path = "Screenshot 2026-06-10 013611.png"
if not os.path.exists(image_path):
    print(f"Image '{image_path}' not found in the root directory. Please ensure Television.png is present to test.")
else:
    print(f"Running inference on {image_path} with confidence threshold 0.20...")
    results = model(image_path, conf=0.20)
    
    # Print the detections
    for r in results:
        boxes = r.boxes
        print(f"Detected {len(boxes)} objects:")
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xywh = box.xywh[0].tolist()
            print(f" - Class: {cls_name}, Confidence: {conf:.2f}, Box (xywh): {xywh}")