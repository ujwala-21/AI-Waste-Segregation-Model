# Local Waste Segregation System (Ultralytics YOLOv11)

This project has been migrated from a cloud-based Roboflow configuration to a **100% local, offline architecture** using **Ultralytics YOLOv11** running on your laptop.

---

## Architecture Overview
1. **Dataset Management**: Roboflow is used only for image annotation and dataset export in YOLO format.
2. **Local Model**: Local training script `train.py` runs on your laptop's CPU (Intel Core Ultra 7) or GPU (if available) and exports `best.pt` and `best.onnx`.
3. **Local Backend**: A Flask server (`app.py`) runs on `127.0.0.1:5000` to perform local YOLOv11 inference (CPU-optimized, confidence threshold `0.20`), reads waste weights and categories from a local Excel file (`waste_dataset.xlsx`), and returns prediction results.
4. **Local Frontend**: `index.html` runs entirely in the web browser, communicating with the local Flask server for multi-object detection, weight estimation, material type display, and bin recommendations.

---

## 1. Local Environment Setup

### Step A: Create a Python Virtual Environment
Open your terminal (PowerShell or Command Prompt) in the project directory and run:
```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# On Windows Command Prompt:
.\venv\Scripts\activate.bat
```

### Step B: Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 2. Dataset Preparation
If you want to train the model yourself:
1. Go to your **Roboflow Dashboard**.
2. Go to the project version and select **Export Dataset**.
3. Choose **YOLOv8** or **YOLOv11** format and select **download zip to computer**.
4. Unzip the downloaded dataset to the following directory:
   `C:\Users\repal\Downloads\Waste Object Detection.v6i.yolov11`
5. The directory should contain the `train/`, `valid/`, `test/` folders, and `data.yaml`.

---

## 3. Local Training Pipeline

The `train.py` script automatically detects available training hardware (running on GPU if NVIDIA CUDA is present, otherwise falling back to CPU) and copy-exports weights.

### Option A: Train YOLO11n (Nano - Default / Recommended for CPU)
This model is lightweight and trains relatively fast on your CPU (~25-50 minutes total):
```bash
python train.py
```

### Option B: Train YOLO11s (Small - Optional / Higher Capacity)
This model offers higher accuracy for 36 classes but takes longer on CPU (~1.25-2.5 hours total):
```bash
python train.py --model s
```

### What happens after training completes?
1. The script prints out evaluation metrics: **Precision**, **Recall**, **F1 Score**, and **mAP50**.
2. It copies the best model weights to the root folder as **`best.pt`**.
3. It exports the trained model to the ONNX format as **`best.onnx`** (using image size 640 and standard augmentations).

---

## 4. Run the Flask Backend
Start the local server. The backend loads `best.pt` (or falls back to `yolo11n.pt` if training hasn't completed yet) and runs local inference on CPU:
```bash
python app.py
```
*The backend runs on `http://127.0.0.1:5000`.*

---

## 5. Launch the Frontend Dashboard
Simply open **`index.html`** in any web browser (Chrome, Edge, or Firefox).
* Ensure your Flask server is running in the background!
* If the Flask server is not running, the frontend will show a warning: `Local backend server is offline. Please start the Flask server.`

---

## 6. Testing Local Inference
You can run a quick standalone test using `test_yolo.py` to verify the local model and library setup:
```bash
python test_yolo.py
```
*This will load the local `best.pt` (or fallback `yolo11n.pt`) and run offline inference on `Television.png` at a confidence threshold of `0.20`, printing the class names and coordinates to the terminal.*

---

## 7. Data Storage & Excel Backup
* The original 12-class dataset configuration has been backed up to **`waste_dataset_backup.xlsx`**.
* The file **`waste_dataset.xlsx`** now contains all 36 classes with default weights and categories appended, leaving the original 12 classes completely unmodified.
* The dynamically generated **`class_names.json`** lists the 36 class names mapped to their model indices.
