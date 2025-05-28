from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import easyocr
import re
from unidecode import unidecode

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

reader = easyocr.Reader(['en'])

def clean_ocr_line(line):
    line = unidecode(line)
    line = line.strip()
    line = re.sub(r'[0oO]', '0', line)
    line = re.sub(r'[1lI|]', '1', line)
    line = re.sub(r'[^a-zA-Z0-9 /.-]', '', line)
    return line.lower()

def clean_ocr_text(text):
    replacements = {
        'O': '0', 'o': '0', 'I': '1', 'l': '1',
        'B': '8', 'S': '5', '|': '1',
        ' ': '', '-': '', '/': '', '\\': ''
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def extract_aadhar_details(img):
    results = {}

    text_lines = reader.readtext(img, detail=0)
    text_lines = [line.strip() for line in text_lines if line.strip()]
    cleaned_lines = [clean_ocr_line(line) for line in text_lines]

    full_text = " ".join([clean_ocr_text(line.lower()) for line in text_lines])

    # Aadhaar number
    match = re.search(r'\b(\d{4}\s*\d{4}\s*\d{4})\b', full_text)
    if match:
        aadhar_num = re.sub(r'\s+', '', match.group(1))
        formatted_aadhar = " ".join([aadhar_num[i:i+4] for i in range(0, 12, 4)])
        results['aadhar_no'] = formatted_aadhar

    # Date of Birth
    dob = None
    date_patterns = [
        r'\b\d{2}[/-]\d{2}[/-]\d{4}\b', 
        r'\b\d{2}\s*[/-]?\s*\d{2}\s*[/-]?\s*\d{4}\b', 
        r'\b\d{8}\b'
    ]
    for idx, line in enumerate(cleaned_lines):
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

    # Gender
    gender = None
    for line in cleaned_lines:
        if any(g in line for g in ['female', 'femle', 'femal']):
            gender = 'Female'
            break
        elif 'male' in line:
            gender = 'Male'
            break
    if gender:
        results['gender'] = gender

    # Name extraction (before DOB)
    def looks_like_name(text):
        if any(char.isdigit() for char in text): return False
        if len(text) < 3 or len(text) > 40: return False
        address_keywords = ['layout', 'street', 'road', 'cross', 'village', 'po', 'ps', 'dist', 'address', 'near', 'area', 'government', 'india', 'republic']
        if any(k in text for k in address_keywords): return False
        return bool(re.fullmatch(r'[a-z. ]+', text))

    name = None
    dob_idx = None
    if 'dob' in results:
        for i, line in enumerate(cleaned_lines):
            if results['dob'] in line:
                dob_idx = i
                break

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

    for line in text_lines:
        if not pan_number:
            match = re.search(pan_regex, line)
            if match:
                pan_number = match.group(1)
        if not dob:
            match = re.search(dob_regex, line)
            if match:
                dob = match.group(1)

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
