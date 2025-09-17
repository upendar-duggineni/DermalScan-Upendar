import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import pandas as pd

# ---------------------------
# Project Header
# ---------------------------
st.set_page_config(page_title="DermalScan: AI Facial Skin Aging", layout="wide")
st.title("DermalScan: AI Facial Skin Aging Detection")
st.markdown(
    """
    **Detect and classify facial aging signs** such as wrinkles, dark spots, puffy eyes, scars, and clear skin.  
    Upload an image to visualize aging signs with annotated bounding boxes and confidence scores.
    """,
    unsafe_allow_html=True
)
st.write("---")

# ---------------------------
# Paths & Configs
# ---------------------------
FACE_PROTO = r"C:\Dermal scan\age_prediction\opencv_face_detector.pbtxt"
FACE_MODEL = r"C:\Dermal scan\age_prediction\opencv_face_detector_uint8.pb"
AGE_PROTO = r"C:\Dermal scan\age_prediction\age_deploy.prototxt"
AGE_MODEL = r"C:\Dermal scan\age_prediction\age_net.caffemodel"
DERMAL_MODEL = r"C:\Dermal scan\mobilenetv2_best_model.h5"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
AGE_BUCKET_CENTERS = np.array([1.0, 5.0, 10.0, 17.5, 28.5, 40.5, 50.5, 80.0])
CLASS_NAMES = ["Acne", "Clear Face", "Dark Spots", "Puffy Eyes", "Scars", "Wrinkles"]

OUTPUT_DIR = r"C:\Dermal scan\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Load Models
# ---------------------------
@st.cache_resource
def load_all_models():
    face_net = cv2.dnn.readNetFromTensorflow(FACE_MODEL, FACE_PROTO)
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    dermal_model = load_model(DERMAL_MODEL)
    return face_net, age_net, dermal_model

face_net, age_net, dermal_model = load_all_models()
st.success("✅ Models loaded successfully!")

# ---------------------------
# Age Correction
# ---------------------------
def corrected_age(age_cont, dermal_label, dermal_conf):
    if dermal_label.lower() == "wrinkles":
        if dermal_conf >= 0.95:
            return max(age_cont, 75.0)
        elif dermal_conf >= 0.90:
            return max(age_cont, 65.0)
        elif dermal_conf >= 0.80:
            return max(age_cont, 55.0)
    return age_cont

# ---------------------------
# Annotate Image
# ---------------------------
def annotate_image(image, confidence_thresh=0.7, border_size=80):
    if image is None:
        return None, None, None

    # Ensure 3-channel BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Add border
    image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size,
                               cv2.BORDER_CONSTANT, value=[0,0,0])
    h, w = image.shape[:2]

    # Face detection
    blob = cv2.dnn.blobFromImage(image, 1.0, (300,300), [104,117,123], True, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    # Detect only first confident face
    face_box = None
    for i in range(detections.shape[2]):
        conf = detections[0,0,i,2]
        if conf > confidence_thresh:
            x1 = int(detections[0,0,i,3]*w)
            y1 = int(detections[0,0,i,4]*h)
            x2 = int(detections[0,0,i,5]*w)
            y2 = int(detections[0,0,i,6]*h)
            face_box = (x1, y1, x2, y2)
            break

    if face_box is None:
        st.warning("⚠ No face detected.")
        return image, None, None

    x1, y1, x2, y2 = face_box
    face = image[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]

    # --- Dermal Prediction ---
    dermal_input = cv2.resize(face, (224,224))
    dermal_input = preprocess_input(dermal_input.astype(np.float32))
    dermal_input = np.expand_dims(dermal_input, axis=0)
    preds = dermal_model.predict(dermal_input, verbose=0)
    class_idx = preds[0].argmax()
    dermal_class = CLASS_NAMES[class_idx]
    dermal_conf = preds[0][class_idx]

    # --- Age Prediction ---
    age_blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(age_blob)
    age_probs = age_net.forward()[0]
    age_cont = float(np.sum(age_probs * AGE_BUCKET_CENTERS))
    age_cont = corrected_age(age_cont, dermal_class, dermal_conf)
    bucket_idx = int(np.argmin(np.abs(AGE_BUCKET_CENTERS - age_cont)))
    age_bucket = AGE_BUCKETS[bucket_idx]
    max_age_conf = float(age_probs.max())

    # Overlay labels
    dermal_label = f"{dermal_class}: {dermal_conf*100:.1f}%"
    age_label = f"Age: {age_bucket} ({age_cont:.1f} yrs)"
    if max_age_conf < 0.45:
        age_label += " [Uncertain]"

    cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(image, dermal_label, (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(image, age_label, (x1, y2+50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Save annotated image
    save_path = os.path.join(OUTPUT_DIR, "annotated_uploaded_image.jpg")
    cv2.imwrite(save_path, image)

    # Return single prediction details
    details = {
        "Dermal Class": dermal_class,
        "Confidence": f"{dermal_conf*100:.1f}%",
        "Predicted Age": age_bucket,
        "Age (Years)": f"{age_cont:.1f}",
        "Age Confidence": f"{max_age_conf*100:.1f}%",
        "Bounding Box": f"(x1={x1}, y1={y1}, x2={x2}, y2={y2})"
    }

    return image, save_path, details

# ---------------------------
# Streamlit UI
# ---------------------------
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg","jpeg","png","webp"])

with st.sidebar.expander("Advanced Options"):
    confidence_thresh = st.slider("Face Detection Confidence", 0.1, 1.0, 0.7, 0.05)
    border_size = st.slider("Image Border Size", 0, 150, 80, 5)

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("❌ Uploaded file is not a valid image.")
    else:
        with st.spinner("Processing..."):
            annotated_image, save_path, details = annotate_image(image, confidence_thresh, border_size)

        st.subheader("Annotated Result")
        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)

        if details:
            st.subheader("Prediction Details")
            st.table(details)

            # --- Single Prediction CSV ---
            df = pd.DataFrame([details])
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📊 Download This Prediction CSV",
                data=csv,
                file_name="prediction_details.csv",
                mime="text/csv",
            )

        if save_path:
            st.success(f"Annotated image saved at: {save_path}")
            st.download_button("📥 Download Annotated Image",
                               data=open(save_path, "rb"),
                               file_name="annotated_image.jpg")
