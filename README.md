# 🧠 Stroke Prediction App (KNN + Streamlit)

A machine learning web app built using Streamlit that predicts the likelihood of stroke using a K-Nearest Neighbors (KNN) model. The app allows users to explore the dataset, train the model, evaluate performance, and predict stroke risk for new patients.

---

## 🚀 Features

- Dataset overview with preview and statistics
- KNN model training with balanced data (SMOTE)
- Model performance evaluation
- Confusion matrix visualization
- Real-time prediction for new patient data

---

## 🧰 Tech Stack

- Python
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)

---

## 📂 Project Structure

stroke-prediction-app/
│
├── app.py
├── healthcare-dataset-stroke-data.csv
├── requirements.txt
└── README.md

---

## 📊 Dataset

The dataset contains patient health information including:

- Age
- Hypertension
- Heart disease
- Average glucose level
- BMI
- Gender
- Marital status
- Work type
- Residence type
- Smoking status
- Stroke label

---

## ⚙️ Installation

1. Clone the repository:

git clone https://github.com/your-username/stroke-prediction-app.git

2. Navigate to the project folder:

cd stroke-prediction-app

3. Install dependencies:

pip install -r requirements.txt

---

## ▶️ Run the App

streamlit run app.py

The app will open in your browser.

---

## 📈 Model Details

- Algorithm: K-Nearest Neighbors (KNN)
- Encoding: One-hot encoding for categorical features
- Class balancing: SMOTE
- Metrics: Precision, Recall, F1-score
- Visualization: Confusion matrix heatmap

---

## 🧪 How to Use

1. Open the app
2. Explore the dataset overview
3. Train the model
4. Enter patient details
5. Get stroke prediction instantly

---

## 📌 Future Improvements

- Add more ML models
- Hyperparameter tuning
- Model saving/loading
- Deployment to cloud

---

## 👨‍💻 Author

Your Name

---

## 📄 License

This project is open-source and free to use.
