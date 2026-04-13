# 🩺 AI-Powered Medical Image Analysis

## 📌 Overview
This project uses Deep Learning to detect **Pneumonia from Chest X-ray images**. It includes model training, evaluation metrics, and a Streamlit-based web interface for real-time predictions.

---

## 🚀 Features
- Pneumonia Detection using Convolutional Neural Networks (CNN)
- Real-time prediction via Streamlit Web App
- Accuracy and Loss Visualization
- Confusion Matrix Analysis
- Classification Report Generation
- Clean and Modular Code Structure

---

## 🗂️ Project Structure

```
AI-Powered-Medical-Image-Analysis/
│── app.py
│── main.py
│── README.md
│── requirements.txt
│
├── src/
│   ├── train.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── predict.py
│
├── model/
│   └── model.keras
│
├── outputs/
│   ├── accuracy.png
│   ├── loss.png
│   ├── confusion_matrix.png
│   ├── normal.png
│   ├── pneunomia.png
│   ├── output1.png
│   ├── output1.1.png
│   ├── streamlite_overview.png
│   ├── report.txt
```

---

## 📊 Model Performance

### 📈 Accuracy Graph
<img src="https://raw.githubusercontent.com/Swetha07062003/AI-Powered-Medical-Image-Analysis/main/outputs/accuracy.png" width="600"/>

### 📉 Loss Graph
<img src="https://raw.githubusercontent.com/Swetha07062003/AI-Powered-Medical-Image-Analysis/main/outputs/loss.png" width="600"/>

### 🔲 Confusion Matrix
<img src="https://raw.githubusercontent.com/Swetha07062003/AI-Powered-Medical-Image-Analysis/main/outputs/confusion_matrix.png" width="600"/>

## 🧠 Prediction Results

### ✅ Normal Case
<img src="https://raw.githubusercontent.com/Swetha07062003/AI-Powered-Medical-Image-Analysis/main/outputs/normal.png" width="400"/>

### ❗ Pneumonia Detected
<img src="https://raw.githubusercontent.com/Swetha07062003/AI-Powered-Medical-Image-Analysis/main/outputs/pneunomia.png" width="400"/>

## 🌐 Streamlit Web Application

### 🖥️ UI Preview
<img src="https://raw.githubusercontent.com/Swetha07062003/AI-Powered-Medical-Image-Analysis/main/outputs/streamlite_overview.png" width="700"/>

## 📄 Additional Outputs

### Prediction Example 1
<img src="https://raw.githubusercontent.com/Swetha07062003/AI-Powered-Medical-Image-Analysis/main/outputs/output1.png" width="500"/>

### Prediction Example 1.1
<img src="https://raw.githubusercontent.com/Swetha07062003/AI-Powered-Medical-Image-Analysis/main/outputs/output1.1.png" width="500"/>

### 📄 Classification Report
Detailed performance metrics (Precision, Recall, F1-Score) are available in:

outputs/report.txt
---


## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Training

```bash
python -m src.train
```

---

## 🌍 Run Web Application

```bash
streamlit run app.py
```

---

## 📁 Dataset
Dataset not included due to size limitations. Download from:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

## ⚠️ Disclaimer
This project is for educational purposes only and should not be used for real-world medical diagnosis.

---

## 🔮 Future Enhancements
- Add Grad-CAM for model explainability  
- Improve accuracy using Transfer Learning (ResNet, VGG16)  
- Deploy application using Streamlit Cloud or AWS  
- Extend to multi-class disease detection  
- Optimize model for real-time clinical usage  

---

## 👩‍💻 Author
Swetha K
