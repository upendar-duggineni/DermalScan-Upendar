# DermalScan-Upendar
AI_Facial Skin Aging  Detection App 

🌟 DermalScan: AI Facial Skin Aging Detection
📖 Overview

DermalScan is a deep learning-powered web application that detects and classifies facial skin aging signs such as wrinkles, dark spots, puffy eyes, scars, acne, and clear skin, while also estimating the age group of the person.

The application integrates:

Dermal condition classification using a MobileNetV2-based model

Face detection using OpenCV DNN

Age prediction using a pretrained Caffe Age Net

Streamlit Web UI for interactive visualization and results export

🚀 Features

🧑‍⚕️ Dermal Condition Detection: Acne, Clear Face, Dark Spots, Puffy Eyes, Scars, Wrinkles

🎯 Age Prediction:

Age Bucket (e.g., (25-32) with confidence %)

Continuous Age Estimate (e.g., 28.4 yrs)

🔄 Wrinkle-based Correction: Adjusts predicted age upwards if wrinkles are detected with high confidence

💻 Web App: Upload images, visualize annotated results with bounding boxes and labels

📥 Export Options: Download annotated images and prediction details in CSV format

🛠️ Tech Stack

Python 3.10+

TensorFlow / Keras – dermal classifier model

OpenCV DNN – face detection & age estimation

Streamlit – interactive web interface

NumPy, Pandas, Matplotlib – data preprocessing, visualization, logging

📂 Project Structure
DermalScan/
│── app.py                 # Streamlit frontend app (main deliverable)
│── train_model.py         # Training script for dermal model
│── models/                # Saved models (mobilenetv2_best_model.h5 etc.)
│── age_prediction/        # Face detector & age net (prototxt, caffemodel)
│── output/                # Annotated results + CSV exports
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation

⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/<your-username>/DermalScan.git
cd DermalScan

2️⃣ Setup Virtual Environment
python -m venv dermal_env
dermal_env\Scripts\activate   # Windows
source dermal_env/bin/activate   # Linux/Mac

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Pretrained Models

Place mobilenetv2_best_model.h5 inside models/

Place the following files inside age_prediction/:

opencv_face_detector.pbtxt

opencv_face_detector_uint8.pb

age_deploy.prototxt

age_net.caffemodel

▶️ Running the Application

Start the Streamlit web app:

streamlit run app.py


Open your browser at http://localhost:8501
 to upload images and view predictions.

📊 Sample Output
🖼 Annotated Image

Annotated bounding box with dermal condition and age predictions.

📋 Prediction Details
Dermal Class: Wrinkles
Confidence: 84.3%
Predicted Age: (48-53)
Age Confidence: 77.1%
Continuous Age: 52.4 yrs
Bounding Box: (x1=296, y1=536, x2=671, y2=1034)

📑 CSV Export Example
Dermal Class,Confidence,Predicted Age,Age Confidence,Continuous Age (yrs),Bounding Box
Wrinkles,84.3%,(48-53),77.1%,52.4,"(x1=296, y1=536, x2=671, y2=1034)"

📈 Future Enhancements

🎨 Enhance UI with custom themes

🧪 Extend support to multi-face detection

📚 Train on larger dataset with more skin conditions

👨‍💻 Contributors

[D.Upendar] – Infosys Springboard Intern

Guided by Infosys Springboard 
