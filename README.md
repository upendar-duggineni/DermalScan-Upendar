# DermalScan: AI Facial Skin Aging Detection App

## 🌟 Overview

DermalScan is a deep learning-powered web application that detects and classifies facial skin aging signs such as wrinkles, dark spots, puffy eyes, scars, acne, and clear skin. It also provides an estimate of the user's age group. The application is built with a modular backend and a clean Streamlit UI, designed for efficient and seamless analysis.

## 🚀 Key Features

* **Dermal Condition Analysis:** Classifies six different skin conditions with percentage confidence scores.
* **Age Prediction:** Provides both a predicted age bucket (e.g., (25-32)) and a continuous age estimate (e.g., 28.4 yrs).
* **Intuitive Web App:** A user-friendly interface that allows for easy image uploads and interactive visualization.
* **Annotated Output:** Displays annotated bounding boxes, labels, and prediction scores directly on the uploaded image.
* **Efficient Backend:** The pipeline is optimized for fast inference, ensuring a processing time of less than 5 seconds per image.
* **Export Options:** Allows users to download the annotated image and a CSV file with detailed prediction logs.

## 🛠️ Tech Stack

| Area          | Tools / Libraries                                        |
|---------------|----------------------------------------------------------|
| **Core Models** | EfficientNetB0, MobileNetV2, OpenCV DNN, Caffe Age Net   |
| **Backend** | Python, TensorFlow / Keras                               |
| **Frontend** | Streamlit                                                |
| **Data & Ops** | NumPy, Matplotlib, Git LFS                                 |

## 📂 Project Structure

DermalScan/
├── app.py                      # Main Streamlit frontend app
├── face_prediction.py          # Backend logic for inference
├── models/                     # All pre-trained models
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── mobilenetv2_best_model.h5
│   ├── opencv_face_detector.pbtxt
│   └── opencv_face_detector_uint8.pb
├── output/                     # Saved annotated images and logs
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation


## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone [https://github.com/your-upendar-duggineni/DermalScan.git](https://github.com/upendar-duggineni/DermalScan-Upendar/edit/main/README.md)
cd DermalScan
2️⃣ Set Up a Virtual Environment
Bash

python -m venv dermal_env
dermal_env\Scripts\activate  # For Windows
source dermal_env/bin/activate # For Linux/Mac
3️⃣ Install Dependencies
Create a requirements.txt file in your project's root directory and install all libraries.

Bash

pip install -r requirements.txt
4️⃣ Add Pretrained Models
Place the necessary model files inside the models/ directory. For large files, you must use Git Large File Storage (Git LFS).

▶️ Running the Application
To start the web app, ensure your virtual environment is active and run the following command in your terminal:

Bash

streamlit run app.py
Open your browser at http://localhost:8501 to use the application.

✅ Project Evaluation
Backend Performance: The application provides a seamless input-to-output flow with a processing time of less than 5 seconds per image.

Usability: The UI is responsive and provides a clean, clear visualization of the annotated outputs.

Documentation: The project includes professional-level documentation, including a README and logged results.

👨‍💻 Contributors
[D.Upendar] – Infosys Springboard Intern

Guided by Infosys Springboard Mentor

