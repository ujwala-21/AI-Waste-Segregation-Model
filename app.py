import os
import tempfile
import uuid
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

@app.route("/")
def index():
    return app.send_static_file("index.html")

# Load local YOLO model (CPU fallback is handled automatically by PyTorch/Ultralytics)
# We check if best.pt exists; if not, we load yolo11n.pt as a fallback so the app starts up
model_path=r"runs\detect\runs\detect\train\weights\best.pt"
print(f"Loading local YOLO model from: {model_path}")
model = YOLO(model_path)

def normalize_name(name):
    if not name:
        return ""
    name = str(name).lower().strip().replace("_", "").replace(" ", "")
    if name in ["papers", "paper"]:
        return "paperbundle"
    if name.endswith("s") and not name.endswith("glass"):
        if name == "leafyvegetables":
            return "leafyvegetable"
        return name[:-1]
    return name

# Material types fallback map
MATERIAL_TYPES = {
    "leafy_vegetables": "Organic",
    "tomato_waste": "Organic",
    "banana_peel": "Organic",
    "gloves": "Latex Rubber",
    "masks": "Polypropylene",
    "paper_bundle": "Cellulose Fiber",
    "plastic_bottle": "Plastic (PET)",
    "television": "Mixed Composites",
    "carrot_peel": "Organic",
    "syringe": "Biohazard Medical",
    "laptop": "E-Waste Circuitry",
    "potato_peel": "Organic",
    "mobile_phone": "Lithium / Metal",
    "egg_shells": "Organic",
    "Onion": "Organic",
    "glass": "Glass",
    "Charger": "Copper / Silicon",
    "keyboard": "Plastic / Electronic",
    "Cardboards": "Cellulose Fiber",
    "apple": "Organic",
    "computer_mouse": "Plastic / Electronic",
    "aluminium_can": "Aluminium",
    "oranges": "Organic",
    "headphones": "Plastic / Electronic",
    "earphones": "Plastic / Copper",
    "battery": "Heavy Metal / Lead",
    "Backpack": "Nylon / Fabric",
    "clothes": "Cotton / Polyester",
    "footware": "Rubber / Leather",
    "kitchen_utensil": "Stainless Steel",
    "spoon": "Stainless Steel / Plastic",
    "remote": "Plastic / Circuitry",
    "chair": "Wood / Plastic",
    "fork": "Stainless Steel / Plastic",
    "Lemon": "Organic",
    "coconut": "Organic"
}

# Physical weights fallback map (in grams)
FALLBACK_WEIGHTS = {
    "leafy_vegetables": 20,
    "tomato_waste": 12,
    "banana_peel": 20,
    "gloves": 7,
    "masks": 5,
    "paper_bundle": 2500,
    "plastic_bottle": 25,
    "television": 5000,
    "carrot_peel": 8,
    "syringe": 12,
    "laptop": 1500,
    "potato_peel": 10,
    "mobile_phone": 180,
    "egg_shells": 5,
    "Onion": 15,
    "glass": 150,
    "Charger": 40,
    "keyboard": 400,
    "Cardboards": 300,
    "apple": 100,
    "computer_mouse": 100,
    "aluminium_can": 15,
    "oranges": 120,
    "headphones": 200,
    "earphones": 30,
    "battery": 20,
    "Backpack": 500,
    "clothes": 300,
    "footware": 400,
    "kitchen_utensil": 250,
    "spoon": 30,
    "remote": 80,
    "chair": 3000,
    "fork": 30,
    "Lemon": 30,
    "coconut": 400
}

def get_material_type_fallback(detected_object):
    if not detected_object:
        return "General Material"
    obj_norm = str(detected_object).lower().strip().replace("_", "").replace(" ", "")
    for k, v in MATERIAL_TYPES.items():
        k_norm = str(k).lower().strip().replace("_", "").replace(" ", "")
        if k_norm == obj_norm:
            return v
    return "General Material"

def get_weight_fallback(detected_object):
    if not detected_object:
        return 15
    obj_norm = str(detected_object).lower().strip().replace("_", "").replace(" ", "")
    for k, v in FALLBACK_WEIGHTS.items():
        k_norm = str(k).lower().strip().replace("_", "").replace(" ", "")
        if k_norm == obj_norm:
            return v
    return 15

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    print("FILES RECEIVED:", request.files)
    print("FORM RECEIVED:", request.form)
    if "image" not in request.files:
        hint = ""
        if "image" in request.form:
            hint = " (field arrived as text, not a file — append a Blob/File to FormData with a filename)"
        return jsonify({"error": "No image file uploaded" + hint}), 400

    image = request.files["image"]
    
    # Save the file to the system temp directory using a safe lock-free filename for Windows compatibility
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.jpg")
    image.save(temp_path)

    # Default confidence threshold is 0.20, but remains configurable via query parameters or form fields
    conf_threshold = request.args.get("confidence", default=0.20, type=float)
    if "confidence" in request.form:
        try:
            conf_threshold = float(request.form["confidence"])
        except ValueError:
            pass

    result = {"predictions": []}
    try:
        # Perform local inference with configured confidence threshold
        yolo_results = model(temp_path, conf=conf_threshold)
        
        predictions = []
        for r in yolo_results:
            boxes = r.boxes
            for box in boxes:
                # Get xywh (center x, center y, width, height)
                xywh = box.xywh[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                
                predictions.append({
                    "x": xywh[0],
                    "y": xywh[1],
                    "width": xywh[2],
                    "height": xywh[3],
                    "confidence": conf,
                    "class": cls_name
                })
        result = {"predictions": predictions}
    except Exception as e:
        print("Inference error:", e)
        result = {"error": str(e), "predictions": []}
    finally:
        # Clean up the temp file after inference
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Read Excel dataset
    try:
        df = pd.read_excel("waste_dataset.xlsx")
    except Exception as e:
        print("Excel read error:", e)
        df = pd.DataFrame()

    weight = 0

    if "predictions" in result and len(result["predictions"]) > 0:
        for pred in result["predictions"]:
            detected_object = pred["class"]
            
            # Fetch from Excel if available (robust normalization search)
            row = pd.DataFrame()
            if not df.empty and "Object_Name" in df.columns:
                detected_object_norm = normalize_name(detected_object)
                for idx, r in df.iterrows():
                    if normalize_name(str(r["Object_Name"])) == detected_object_norm:
                        row = df.iloc[[idx]]
                        break
            
            if not row.empty:
                item_weight = int(row.iloc[0]["Weight_g"]) if "Weight_g" in row.columns and not pd.isna(row.iloc[0]["Weight_g"]) else get_weight_fallback(detected_object)
                item_category = str(row.iloc[0]["Category"]) if "Category" in row.columns and not pd.isna(row.iloc[0]["Category"]) else "General Waste"
                # Fallback to local material type mapping if not in Excel
                item_material = str(row.iloc[0]["Material_Type"]) if "Material_Type" in row.columns and not pd.isna(row.iloc[0]["Material_Type"]) else get_material_type_fallback(detected_object)
            else:
                # Fallback normalized mapping for all 36 classes
                item_weight = get_weight_fallback(detected_object)
                
                detected_object_norm = normalize_name(detected_object)
                wet_norms = ["bananapeel", "carrotpeel", "leafyvegetable", "tomatowaste", "potatopeel", "eggshell", "onion", "apple", "orange", "lemon", "coconut"]
                dry_norms = ["plasticbottle", "paperbundle", "glass", "cardboard", "aluminiumcan", "backpack", "clothe", "footware", "kitchenutensil", "spoon", "chair", "fork"]
                medical_norms = ["syringe", "mask", "glove"]
                electronic_norms = ["laptop", "mobilephone", "television", "charger", "keyboard", "computermouse", "headphone", "earphone", "battery", "remote"]
                
                if detected_object_norm in wet_norms:
                    item_category = "Wet Waste"
                elif detected_object_norm in dry_norms:
                    item_category = "Dry Waste"
                elif detected_object_norm in medical_norms:
                    item_category = "Medical Waste"
                elif detected_object_norm in electronic_norms:
                    item_category = "Electronic Waste"
                else:
                    item_category = "General Waste"
                
                item_material = get_material_type_fallback(detected_object)
                
            pred["weight_g"] = item_weight
            pred["category"] = item_category
            pred["material_type"] = item_material
            weight += item_weight

    result["estimated_weight"] = weight

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)