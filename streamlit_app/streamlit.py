import streamlit as st
import requests
import boto3
from PIL import Image
import io
import re

FLASK_API_URL = "http://13.126.26.34:1818"



def textract_image(image_bytes, doc_type):
    session = boto3.Session(profile_name="wizzgeeks_profile")
    textract = session.client('textract', region_name='ap-south-1')
    response = textract.detect_document_text(Document={'Bytes': image_bytes})
    print(response)
    # st.write("Textract Response:", response)

    if "Blocks" not in response:
        return {"error": "No Blocks found in Textract response"}

    text_lines = [item.get("Text", "") for item in response["Blocks"] if item.get("BlockType") == "LINE"]

    print("\n--- Textract detected lines ---")
    for line in text_lines:
        print(line)
    print("--- End of detected lines ---\n")

    full_text = " ".join(text_lines).replace("\n", " ")

    if doc_type == "PAN":
      pan_number = None
      name = None
      dob = None

      full_text = " ".join(text_lines).replace("\n", " ")



      pan_match = re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', full_text)
      if pan_match:
        pan_number = pan_match.group(0)

        for idx, line in enumerate(text_lines):
            match = re.search(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b', line)
            if match:
                dob = match.group(0)
                dob_idx = idx
                break
        for i, line in enumerate(text_lines):
            if "name" in line.lower():

                if i + 1 < len(text_lines):
                    possible_name = text_lines[i + 1].strip()
                    if len(possible_name.split()) <= 4:
                        name = possible_name.title()
                        break


        if not name and dob:
            for i in range(dob_idx - 1, max(dob_idx - 4, -1), -1):
                candidate = text_lines[i].strip()
                if re.match(r"^[A-Z ]{3,}$", candidate):
                    name = candidate.title()
                    break

        return {
            "name": name if name else "NULL",
            "dob": dob if dob else "NULL",
            "pan_number": pan_number if pan_number else "NULL"
        }


    else: 
        aadhar_no_match = re.search(r'(\d{4}\s?\d{4}\s?\d{4})', full_text)

        dob = None
        for idx, line in enumerate(text_lines):
            if any(k in line.lower() for k in ["dob", "date of birth", "जन्म"]):
                match = re.search(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b', line)
                if match:
                    dob = match.group(0)
                    break
                elif idx + 1 < len(text_lines):
                    match = re.search(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b', text_lines[idx + 1])
                    if match:
                        dob = match.group(0)
                        break

        dob_idx = None
        gender_idx = None
        for idx, line in enumerate(text_lines):
            line_lower = line.lower()
            if any(k in line_lower for k in ["dob", "date of birth", "जन्म"]) and re.search(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b', line):
                dob_idx = idx
            if "male" in line_lower or "female" in line_lower:
                gender_idx = idx

        address_keywords = ["layout", "street", "road", "cross", "village", "po", "ps", "dist", "address", "near", "area"]
        name = None

        # 👇 New logic: Get name before DOB line
        if dob_idx is not None:
            for i in range(dob_idx - 1, max(dob_idx - 5, -1), -1):
                candidate = text_lines[i].strip()
                if any(k in candidate.lower() for k in address_keywords + ["government of india", "republic of india"]):
                    continue  # Skip address or header lines
                if re.fullmatch(r"[A-Za-z. ]{3,}", candidate) and len(candidate.split()) <= 4:
                    name = candidate
                    break

        # 🔁 Fallback logic if name not found
        if not name:
            for line in text_lines:
                line_clean = line.strip()
                if any(k in line_clean.lower() for k in address_keywords + ["government of india", "republic of india"]):
                    continue
                if re.fullmatch(r"[A-Za-z. ]{3,}", line_clean) and len(line_clean.split()) <= 4:
                    name = line_clean
                    break

        gender = None
        for line in text_lines:
            lower_line = line.lower()
            if "female" in lower_line:
                gender = "Female"
                break
            elif "male" in lower_line:
                gender = "Male"
                break

        def safe_val(v):
            return v if v else "NULL"

        return {
            "name": safe_val(name),
            "dob": safe_val(dob),
            "gender": safe_val(gender),
            "aadhar_no": safe_val(aadhar_no_match.group(0) if aadhar_no_match else None)
        }


st.set_page_config(page_title="OCR ID Extractor", layout="wide")

st.title("🪪 Aadhar & PAN Extractor")
st.markdown(
    "Upload an image of your Aadhar or PAN card to extract details using **Local YOLO+EasyOCR** and **AWS Textract** simultaneously."
)

option = st.selectbox("Select Document Type", ["Aadhar", "PAN"])
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    bytes_data = uploaded_file.getvalue()
    image_bytes_io = io.BytesIO(bytes_data)

    if st.button("Extract with Both Engines"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔍 Local (YOLO + EasyOCR)")
            try:
                endpoint = f"{FLASK_API_URL}/extract_aadhar" if option == "Aadhar" else f"{FLASK_API_URL}/extract_pan"

                image_bytes_io.seek(0)
                files = {'image': (uploaded_file.name, image_bytes_io, uploaded_file.type)}

                with st.spinner("Running Local OCR..."):
                    response = requests.post(endpoint, files=files)
                    response.raise_for_status()
                    result = response.json()

                if result.get("success"):
                    st.success("Local OCR Success")
                    st.json(result["data"])
                else:
                    st.error(f"Local OCR Failed: {result.get('error', 'Unknown Error')}")
            except Exception as e:
                st.error(f"Local OCR Error: {str(e)}")

        with col2:
            st.subheader("🧠 AWS Textract")
            try:
                with st.spinner("Calling AWS Textract..."):
                    textract_result = textract_image(bytes_data, option)
                st.success("Textract Success")
                st.json(textract_result)
            except Exception as e:
                st.error(f"AWS Textract Error: {str(e)}")
