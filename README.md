# 🌿 Plant Disease Detection using CNN

## 📝 Project Overview
Plant diseases can severely impact crop yield and food security. Early and accurate detection of plant diseases is crucial for timely intervention. This project utilizes **deep learning-based image classification** to detect diseases in plant leaves. The system is trained on the **PlantVillage dataset** and uses **CNN models (DenseNet121, VGG16, and InceptionV3)** to classify images of leaves and predict diseases.

A **bagging ensemble** of these three models is implemented to enhance accuracy and ensure robust predictions. The model is trained to classify diseases in **four plant types**:
- ✅ **Tomato** 🍅
- ✅ **Potato** 🥔
- ✅ **Corn** 🌽
- ✅ **Apple** 🍏

The project is designed to help **farmers, agricultural experts, and researchers** detect plant diseases early, enabling quick preventive measures and improving crop health.

---

## 🚀 Features
✔️ **Upload an Image:** Users can upload a plant leaf image via the web interface.  
✔️ **Deep Learning-based Classification:** The system predicts the disease using **CNN models** trained on labeled plant leaf images.  
✔️ **Bagging Ensemble Approach:** A combination of **VGG16, InceptionV3, and DenseNet121** improves prediction accuracy.  
✔️ **Fast and Efficient Predictions:** With **Flask as the backend** and **React for the frontend**, the system provides quick and user-friendly results.  
✔️ **Scalable and Deployable:** Can be extended to include more plant species and integrated into mobile or web-based agricultural solutions.  

---

## 🏗️ Tech Stack
### 🔹 **Machine Learning & Deep Learning:**
- **TensorFlow / Keras** – For training CNN models
- **VGG16, InceptionV3, and DenseNet** – Pre-trained models for feature extraction
- **Bagging Ensemble Learning** – To improve final prediction accuracy

### 🔹 **Backend:**
- **Flask** – Lightweight and fast API to handle image processing and model inference
- **Python** – Core programming language for model training and backend integration

### 🔹 **Frontend:**
- **React.js** – Interactive and dynamic UI for users to upload images and get results
- **Axios** – To communicate between the frontend and backend

---

## 📂 Dataset
We used the **PlantVillage Dataset** for training the models. This dataset contains **over 50,000** labeled plant leaf images across multiple species.  
🔗 **Download Here:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)  

---

## 🎯 Goals & Impact
🌱 **Early Detection:** Identifying plant diseases at an early stage to reduce crop losses.  
🌍 **Agricultural Support:** Assisting farmers in making informed decisions about plant health.  
📈 **Scalability:** Extending the model to support more plant species and diseases in future updates.  

---

## 🛠 Installation & Usage

### 🔧 1. Clone the Repository
```bash
git clone https://github.com/rahuldewangan05/Plant-Disease-Detection.git  
cd plant-disease-detection
```

### 🖥️ 2. Install Dependencies  
#### Backend (Flask)
```bash
cd backend  
pip install -r requirements.txt  
python app.py
```

#### Frontend (React)
```bash
cd frontend  
npm install  
npm start
```

### 📸 3. Upload a Plant Leaf Image
- Open the web interface  
- Upload an image of a plant leaf  
- Get instant disease classification  

---

## 🔥 Future Enhancements
- 🏷️ **Support for More Plant Species**  
- 📊 **Explainable AI for Disease Insights**  
- 📱 **Mobile App Deployment**  
- ☁️ **Cloud Integration for Large-Scale Deployment**  

