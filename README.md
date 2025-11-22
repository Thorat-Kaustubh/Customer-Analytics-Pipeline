<h1 align="center">🛒 Customer Analytics Pipeline — Segmentation & CLV Prediction</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Project-Customer_Analytics-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Tech-Python_|_Sklearn_|_XGBoost_|_Streamlit-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=for-the-badge">
</p>

---

## 📘 Overview

The **Customer Analytics Pipeline** is a complete, end-to-end machine learning workflow that converts raw customer transaction data into:

- 🎯 Customer segmentation  
- 📊 Behavioral insights  
- 💰 90-day Customer Lifetime Value (CLV) predictions  
- 📈 Interactive dashboards  
- 📄 Automated multi-page PDF business reports  

It follows a **modular, production-ready architecture** and is built for:

✔ Real business deployments  
✔ MNC data science & engineering interviews  
✔ Portfolio projects  
✔ Enterprise analytics  

---

## 🚀 Key Features

### 🔹 1. RFM Feature Engineering  
Automatically generates:
- Recency  
- Frequency  
- Monetary Total  
- Monetary Average  
- Monetary Variance  

---

### 🔹 2. Smart Customer Segmentation (K-Means)
Creates actionable customer groups:
- 🏆 Champions  
- 💎 Loyal  
- 🌱 Potential Loyalists  
- ⚠️ At-Risk  
- ❌ Lost  

---

### 🔹 3. CLV Prediction  
Trains and compares **7 regression models**:

| Model | Included |
|-------|----------|
| Linear Regression | ✔ |
| Ridge | ✔ |
| Lasso | ✔ |
| Random Forest | ✔ |
| Extra Trees | ✔ |
| Gradient Boosting | ✔ |
| XGBoost | ✔ |

➡️ **Best-performing model is automatically selected using MAE.**

---

### 🔹 4. Modern Streamlit Dashboard  
Includes:
- KPI Cards  
- Segment Distribution  
- RFM Scatter Visuals  
- CLV Distribution  
- Model Comparison  
- Customer Explorer  
- CSV Export  

---

### 🔹 5. Automated PDF Reporting  
Exports a professional multi-page PDF containing:
- Executive Summary  
- Segment Insights  
- Model Comparison Tables  
- CLV Predictions  
- Visualizations  

---

## 🖼️ Dashboard Preview

![Customer-Analytics-Pipeline](image1.png)

![Customer-Analytics-Pipeline](image2.png)

![Customer-Analytics-Pipeline](image3.png)

![Customer-Analytics-Pipeline](image4.png)


## ⚙️ Installation

Follow the steps below to set up the project locally:

1️⃣ Clone the Repository
git clone https://github.com/YOUR_USERNAME/customer-analytics-pipeline.git
cd customer-analytics-pipeline

2️⃣ Create a Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate

Mac / Linux
python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Your Dataset

Place your raw input file (CSV/Excel) inside:

data/raw/


For example:

data/raw/online_retail_II.csv

5️⃣ Run the End-to-End Pipeline

This performs:
✔ RFM Feature Engineering
✔ KMeans Segmentation
✔ CLV Label Generation
✔ Training 7 Regression Models
✔ Model Comparison
✔ Exporting Visuals/Models

python main_pipeline.py

6️⃣ Run the Streamlit Dashboard

Launch the modern UI:

streamlit run app.py

7️⃣ Generate PDF Report

Inside the Streamlit app sidebar:
Click:

📄 Generate PDF Report

A full multi-page analytics report will download automatically.

## ✉️ Contact

If you have questions, feedback, or collaboration ideas, feel free to reach out:

Kaustubh Thorat
📧 Email: kaustubhthorat07@gmail.com


