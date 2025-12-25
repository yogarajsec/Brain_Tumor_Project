# 🧠 Brain Tumor Classification using CNN & Flask

This project is a Brain Tumor Classification system developed using Convolutional Neural Networks (CNN) and deployed using Flask as a web application.  
Users can upload an MRI image and the system predicts the type of brain tumor.

---

## 🚀 Features
- CNN-based Brain Tumor Classification
- MRI image upload and prediction
- Flask web interface
- Train model locally
- Test images without retraining
- Works on CPU and GPU
- Simple UI for students and mini projects

---

## 🛠️ Technologies Used
- Python 3.9
- TensorFlow 2.9.1
- Keras
- Flask
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- HTML & CSS
- VS Code

---

## 📂 Project Structure

Brain_Tumor_Project/
│── app.py  
│── main.py  
│── Brain_Tumors.h5 (ignored in GitHub)  
│── class_names.npy  
│── dataset/  
│   ├── Training/  
│   └── Testing/  
│── templates/  
│   └── index.html  
│── static/  
│   └── uploads/  
│── venv/  
│── .gitignore  
│── README.md  

---

## 📥 Dataset

The dataset contains MRI images categorized into different brain tumor classes.

Example folder structure:

dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/

Dataset source:
- Kaggle – Brain Tumor MRI Dataset

---

## ⚙️ Installation & Setup

1. Clone the repository

git clone https://github.com/your-username/Brain_Tumor_Project.git  
cd Brain_Tumor_Project

2. Create virtual environment

python -m venv venv  
venv\Scripts\activate

3. Install dependencies

pip install tensorflow==2.9.1 flask opencv-python numpy matplotlib pillow scikit-learn

---

## 🧠 Train the Model

python main.py  

Choose:
1 → Train Model

This will generate:
- Brain_Tumors.h5
- class_names.npy

---

## 🧪 Test Image (Without Training)

python main.py  

Choose:
2 → Test Image

Enter the image path when prompted.

---

## 🌐 Run Flask Web Application

python app.py  

Open browser and go to:
http://127.0.0.1:5000/

Upload MRI image and get prediction result.

---

## ⚠️ Important Notes
- Brain_Tumors.h5 is not uploaded to GitHub due to size limit
- Dataset is ignored in GitHub
- Train the model locally before testing

---

## 👨‍🎓 Academic Use
This project is suitable for:
- Mini Project
- Final Year Project
- Deep Learning Lab
- Machine Learning Lab

---

## ✨ Author
Yogaraj  
Computer Science Student

---

## 📜 License
This project is created for educational purposes only.
