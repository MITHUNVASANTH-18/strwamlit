from flask import Flask, request, jsonify
import cv2
import numpy as np
import easyocr
import os
import re
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

reader = easyocr.Reader(['en'])

net_aadhar = cv2.dnn.readNet("yolov3last2.weights", "yolov3.cfg")
aadhar_classes = ["name", "dob", "gender", "aadhar_no"]
layer_names = net_aadhar.getLayerNames()
output_layers_aadhar = [layer_names[i - 1] for i in net_aadhar.getUnconnectedOutLayers()]
colors_aadhar = np.random.uniform(0, 255, size=(len(aadhar_classes), 3))


import re
from unidecode import unidecode

def clean_ocr_line(line):
    line = unidecode(line)
    line = line.strip()
    line = re.sub(r'[0oO]', '0', line)
    line = re.sub(r'[1lI|]', '1', line)
    line = re.sub(r'[^a-zA-Z0-9 /.-]', '', line)
    return line.lower()

def clean_ocr_text(text):
    replacements = {
        'O': '0',
        'o': '0',
        'I': '1',
        'l': '1',
        'B': '8',
        'S': '5',
        '|': '1',
        ' ': '',
        '-': '',
        '/': '',
        '\\': ''
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text



def extract_aadhar_details(img):
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416),
                                 mean=(0, 0, 0), swapRB=True, crop=False)
    net_aadhar.setInput(blob)
    outs = net_aadhar.forward(output_layers_aadhar)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = max(0, int(center_x - w / 2))
                y = max(0, int(center_y - h / 2))
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    results = {}

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = aadhar_classes[class_ids[i]]
        crop = img[y:y + h, x:x + w]
        text = reader.readtext(crop, detail=0)
        results[label] = " ".join(text)


    text_lines = reader.readtext(img, detail=0)
    text_lines = [line.strip() for line in text_lines if line.strip()]
    print('text_lines',text_lines)
    cleaned_lines = [clean_ocr_line(line) for line in text_lines]
    print('cleaned_lines',cleaned_lines)
    if 'aadhar_no' not in results or not results['aadhar_no']:
        full_text = " ".join([clean_ocr_text(line.lower()) for line in text_lines])

        match = re.search(r'\b(\d{4}\s*\d{4}\s*\d{4})\b', full_text)
        if match:
            aadhar_num = re.sub(r'\s+', '', match.group(1))
            formatted_aadhar = " ".join([aadhar_num[i:i+4] for i in range(0, 12, 4)])
            results['aadhar_no'] = formatted_aadhar

    if 'dob' not in results or not results['dob']:
        dob = None
        date_patterns = [
            r'\b\d{2}[/-]\d{2}[/-]\d{4}\b', 
            r'\b\d{2}\s*[/-]?\s*\d{2}\s*[/-]?\s*\d{4}\b', 
            r'\b\d{8}\b' 
        ]
        for idx, line in enumerate(cleaned_lines):
            if re.search(r'\b\d{4}\s*\d{4}\s*\d{4}\b', line):
                continue

            for pattern in date_patterns:
                match = re.search(pattern, line)
                if match:
                    dob_raw = match.group(0).replace(' ', '').replace('-', '/')
                    if re.fullmatch(r'\d{8}', dob_raw):
                        dob = f"{dob_raw[0:2]}/{dob_raw[2:4]}/{dob_raw[4:8]}"
                    else:
                        dob = dob_raw
                    break
            if dob:
                break
        if dob:
            results['dob'] = dob



    if 'gender' not in results or not results['gender']:
        gender = None
        for line in cleaned_lines:
            if any(f in line for f in ['female', 'femle', 'femal']):
                gender = 'Female'
                break
            elif 'male' in line:
                gender = 'Male'
                break
        if gender:
            results['gender'] = gender

    name = None
    dob_idx = None
    if 'dob' in results:
        for i, line in enumerate(cleaned_lines):
            if results.get('dob') and results['dob'] in line:
                dob_idx = i
                break

    address_keywords = ['layout', 'street', 'road', 'cross', 'village', 'po', 'ps',
                        'dist', 'address', 'near', 'area', 'government', 'india', 'republic']

    def looks_like_name(text):
        if any(char.isdigit() for char in text):
            return False
        if len(text) < 3 or len(text) > 40:
            return False
        if any(k in text for k in address_keywords):
            return False
        return bool(re.fullmatch(r'[a-z. ]+', text))

    if dob_idx is not None:
        for i in range(dob_idx - 1, max(-1, dob_idx - 6), -1):
            candidate = cleaned_lines[i]
            if looks_like_name(candidate):
                name = candidate.title()
                break

    if not name:
        for line in cleaned_lines:
            if looks_like_name(line):
                name = line.title()
                break

    if name:
        results['name'] = name

    return results



def extract_pan_details(text_lines):
    pan_number = None
    name = None
    dob = None

    pan_regex = r'\b([A-Z]{5}[0-9]{4}[A-Z])\b'
    dob_regex = r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b'

    # PAN & DOB detection
    for line in text_lines:
        if not pan_number:
            match = re.search(pan_regex, line)
            if match:
                pan_number = match.group(1)

        if not dob:
            match = re.search(dob_regex, line)
            if match:
                dob = match.group(1)

    # Detect name from line after "Name"
    for i, line in enumerate(text_lines):
        if "name" in line.lower() and i + 1 < len(text_lines):
            next_line = text_lines[i + 1].strip()
            if re.match(r'^[A-Z .]{3,}$', next_line.upper()):
                name = next_line.title()
                break

    return {
        "pan_number": pan_number or "NULL",
        "name": name or "NULL",
        "dob": dob or "NULL"
    }
@app.route('/extract_aadhar', methods=['POST'])
def extract_aadhar():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400
  
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join("temp_" + filename)
    file.save(filepath)
    print(file)

    img = cv2.imread(filepath)
    if img is None:
        os.remove(filepath)
        return jsonify({'success': False, 'error': 'Invalid image'}), 400

    results = extract_aadhar_details(img)
    os.remove(filepath)

    return jsonify({'success': True, 'data': results})


@app.route('/extract_pan', methods=['POST'])
def extract_pan():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join("temp_" + filename)
    file.save(filepath)

    result = reader.readtext(filepath, detail=0)
    os.remove(filepath)

    pan_data = extract_pan_details(result)

    if pan_data['pan_number'] is None:
        return jsonify({'success': False, 'error': 'PAN number not found'}), 404

    return jsonify({'success': True, 'data': pan_data})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1818, debug=True)
