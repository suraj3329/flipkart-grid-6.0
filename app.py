from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import cv2
import numpy as np
import os
from keras.models import load_model
from pyzbar.pyzbar import decode
from datetime import datetime
import pandas as pd
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
from PIL import Image
import sqlite3

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Upload folder for files
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize freshness detection model
MODEL_PATH = 'model.pkl/archive (3)fruit_veg_model.keras'
model = load_model(MODEL_PATH)

categories = [
    'Apple(1-5)', 'Apple(5-10)', 'Banana(1-5)', 'Banana(5-10)', 'Banana(10-15)', 'Banana(15-20)',
    'Carrot(1-2)', 'Carrot(3-4)', 'Tomato(1-5)', 'Carrot(5-6)', 'Tomato(5-10)', 'Tomato(10-15)', 'Expired'
]
shelf_life_map = {
    'Apple(1-5)': (1-5), 'Apple(5-10)': (5-10), 'Banana(1-5)': (1-5), 'Banana(5-10)': (5-10),
    'Banana(10-15)': (10-15), 'Banana(15-20)': (15-20), 'Carrot(1-2)': (1-2), 'Carrot(3-4)': (3-4),
    'Carrot(5-6)': (5-6), 'Tomato(1-5)': (1-5), 'Tomato(5-10)': (5-10), 'Tomato(10-15)': (10-15), 'Expired': (0-0)
}
predictions_log = []

# Initialize Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Z4KHc9oZDXD8TZSddTuv"
)
MODEL_ID = "grocery-dataset-q9fj2/5"

# Preprocess image for freshness detection
def preprocess_image(img):
    img_resized = cv2.resize(img, (300, 300))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized

# Predict shelf life
def predict_shelf_life(img):
    img_preprocessed = preprocess_image(img)
    pred = model.predict(img_preprocessed)
    class_idx = np.argmax(pred)
    class_label = categories[class_idx]
    return class_label, shelf_life_map[class_label]

#Routes
@app.route('/')
def home():
    return render_template('base.html')

@app.route('/brand-page')
def brand_route():
    return render_template('brand.html')

@app.route('/freshness-page')
def freshness_route():
    return render_template('freshness.html')

@app.route('/barcode-page')
def barcode_route():
    return render_template('barcode.html')

@app.route('/freshness', methods=['GET', 'POST'])
def freshness_detection():
    if request.method == 'POST':
        image_file = request.files['image']
        if not image_file:
            return jsonify({'error': 'No image uploaded'}), 400

        image_np = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        class_label, days_left = predict_shelf_life(image_np)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        predictions_log.append({
            'Timestamp': timestamp,
            'Prediction': class_label,
            'Shelf Life (Days)': days_left
        })

        return jsonify({'label': class_label, 'days_left': days_left})
    return render_template('freshness.html')

@app.route('/brand', methods=['GET', 'POST'])
def brand_detection():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            result = CLIENT.infer(filepath, model_id=MODEL_ID)
            predictions = result.get("predictions", [])

            if not predictions:
                return jsonify({"error": "No predictions found"}), 200

            image = cv2.imread(filepath)
            for pred in predictions:
                x = int(pred["x"] - pred["width"] / 2)
                y = int(pred["y"] - pred["height"] / 2)
                w = int(pred["width"])
                h = int(pred["height"])
                label = f"{pred['class']} ({pred['confidence']:.2f})"
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            output_filename = f"annotated_{file.filename}"
            output_path = os.path.join(UPLOAD_FOLDER, output_filename)
            cv2.imwrite(output_path, image)

            return jsonify({
                "predictions": predictions,
                "annotated_image_url": f"/get_result/{output_filename}"
            })
        except Exception as e:
            return jsonify({"error": f"Processing error: {str(e)}"}), 500
    return render_template('brand.html')



# Initialize database
def initialize_database():
    conn = sqlite3.connect("barcode_data.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS barcodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            barcode_data TEXT UNIQUE,
            product_name TEXT,
            mfg_date TEXT,
            exp_date TEXT,
            life_expectancy INTEGER
        )
    ''')
    conn.commit()
    conn.close()

# Add data to the database
def add_to_database(barcode_data, product_name, mfg_date, exp_date):
    conn = sqlite3.connect("barcode_data.db")
    cursor = conn.cursor()
    life_expectancy = calculate_life_expectancy( exp_date)
    try:
        cursor.execute('''
            INSERT INTO barcodes (barcode_data, product_name, mfg_date, exp_date, life_expectancy)
            VALUES (?, ?, ?, ?, ?)
        ''', (barcode_data, product_name, mfg_date, exp_date, life_expectancy))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()

# Calculate life expectancy
from datetime import datetime

def calculate_life_expectancy(exp_date):
    # Use the current date instead of the manufacturing date
    today = datetime.now()
    exp_date = datetime.strptime(exp_date, "%Y-%m-%d")
    
    # Calculate life expectancy in days
    life_expectancy = (exp_date - today).days
    return max(life_expectancy, 0)  # Ensure no negative values


# Fetch data from the database
def fetch_from_database(barcode_data):
    conn = sqlite3.connect("barcode_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT product_name, mfg_date, exp_date, life_expectancy FROM barcodes WHERE barcode_data = ?", (barcode_data,))
    result = cursor.fetchone()
    conn.close()
    return result

# Endpoint to process uploaded images
# Process uploaded images
@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = Image.open(file.stream)

    barcodes = decode(image)
    results = []

    for barcode in barcodes:
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type

        if barcode_type == "QRCODE":
            # Handle QR codes
            if "EXP" in barcode_data:
                product_info, exp_date = barcode_data.split("|EXP")
                results.append({
                    "type": "QR Code",
                    "data": barcode_data,
                    "product_name": product_name,
                    "mfg_date": mfg_date,
                    "exp_date": exp_date,
                    "life_expectancy": life_expectancy
                })
            else:
                results.append({
                    "type": "QR Code",
                    "data": barcode_data,
                    "product_info": "Unknown",
                    "expiration_date": "Unknown"
                })
        else:
            # Handle barcodes
            product_info = fetch_from_database(barcode_data)
            if product_info:
                product_name, mfg_date, exp_date, life_expectancy = product_info
                results.append({
                    "type": "Barcode",
                    "data": barcode_data,
                    "product_name": product_name,
                    "mfg_date": mfg_date,
                    "exp_date": exp_date,
                    "life_expectancy": life_expectancy
                })
            else:
                # Default values for unknown barcodes
                results.append({
                    "type": "Barcode",
                    "data": barcode_data,
                    "product_name": "Unknown",
                    "mfg_date": "Unknown",
                    "exp_date": "Unknown",
                    "life_expectancy": "Unknown"
                })

    return jsonify(results)

# Add sample data
initialize_database()
add_to_database("123456789012", "Milk", "2024-11-31", "2025-12-31")
add_to_database("987654321098", "Bread", "2024-11-15", "2025-01-15")
add_to_database("117117551919", "Eggs", "2024-10-01", "2026-10-01")
add_to_database("711212827922", "Butter", "2024-09-30", "2025-04-30")
add_to_database("243394557477", "Lays chips", "2024-08-03", "2025-12-13")
add_to_database("241314517417", "Lays chips", "2024-08-03", "2025-12-23")
add_to_database("273544559697", "Colgate", "2024-08-03", "2025-12-03")
add_to_database("224344856677", "Colgate", "2024-08-03", "2025-12-13")
add_to_database("243344756677", "Colgate", "2024-09-03", "2025-12-23")
add_to_database("623644456677", "Colgate", "2024-09-03", "2025-12-09")

@app.route('/get_result/<filename>')
def get_result(filename):
    try:
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=False)
    except Exception as e:
        return jsonify({"error": f"File not found: {str(e)}"}), 404

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)


