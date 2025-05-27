import cv2
import numpy as np
import pytesseract
import re
import os
import json
import time
import spacy
from pathlib import Path

four_points = []

def draw_circle(event, x, y, flags, param):
    global four_points
    if event == cv2.EVENT_LBUTTONDOWN:
        four_points.append([x, y])
        cv2.circle(param, (x, y), 5, (255, 0, 0), -1)

def image_processing(img, address=False):
    global four_points
    four_points = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_copy = img.copy()

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle, img)

    while len(four_points) != 4:
        cv2.imshow('image', img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

    src_pts = np.float32(four_points)
    dst_pts = np.float32([[0, 0], [1500, 0], [0, 400], [1500, 400]]) if address else np.float32([[0, 0], [850, 0], [0, 550], [850, 550]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(img_copy, matrix, (1500, 400) if address else (850, 550))

    thresh = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV if address else cv2.THRESH_BINARY, 55 if address else 77, 17)

    if address:
        kernel = np.ones((3, 2), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=2)

    return thresh

def get_address(img):
    global four_points
    four_points = []
    thresh = image_processing(img, address=True)
    text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 4 --oem 3')
    cleaned = os.linesep.join([s for s in text.splitlines() if s])
    return cleaned

def get_values(img):
    global four_points
    four_points = []
    NER = spacy.load("en_core_web_sm")
    thresh = image_processing(img)

    text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 4 --oem 3')
    doc = NER(text)

    regex_name = next(([w.text for w in doc.ents if w.label_ == "PERSON"]), None)
    if not regex_name:
        regex_name = re.findall("[A-Z][a-z]+", text)

    regex_gender = re.findall("MALE|FEMALE|male|female|Male|Female", text)
    regex_gender = regex_gender[0] if regex_gender else None

    regex_dob = re.findall(r"\d{2}/\d{2}/\d{4}", text)
    if not regex_dob:
        regex_dob = re.findall(r"\d{4}", text)
    regex_dob = regex_dob[0] if regex_dob else None

    regex_mobile_number = re.findall(r"\d{10}", text)
    regex_mobile_number = regex_mobile_number[0] if regex_mobile_number else None

    regex_aadhaar_number = re.findall(r"\d{4} \d{4} \d{4}", text)
    regex_aadhaar_number = regex_aadhaar_number[0] if regex_aadhaar_number else None

    return regex_name, regex_gender, regex_dob, regex_mobile_number, regex_aadhaar_number

def send_to_json(data):
    time_sec = str(time.time()).replace(".", "_")
    json_data = {time_sec: data}
    path = Path(f"aadhaar_info_{time_sec}.json")
    with open(path, "w") as f:
        json.dump(json_data, f, indent=4)
    return str(path)
