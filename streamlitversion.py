import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Stroke Prediction App", layout="wide")

st.title("🧠 Stroke Prediction App (KNN Only)")

# -------------------
# Load Data
# -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df["bmi"] = df["bmi"].fillna(df["bmi"].mean())
    return df

df = load_data()

# -------------------
# Sidebar Navigation
# -------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "Train Model", "Predict New Patient"])

# -------------------
# Dataset Overview Tab
# -------------------
if page == "Dataset Overview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Info")
    buffer = df.info()
    st.text(buffer)

    st.subheader("Dataset Description")
    st.dataframe(df.describe())

# -------------------
# Train Model Tab
# -------------------
elif page == "Train Model":
    st.subheader("Train K-Nearest Neighbors Model")

    # Feature Engineering
    x = df[["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]]
    y = df["stroke"]

    categorical_features = ["gender","ever_married","work_type","Residence_type","smoking_status"]
    le = OneHotEncoder(sparse_output=False, drop="first")
    en = le.fit_transform(df[categorical_features])
    en_cols = le.get_feature_names_out(categorical_features)
    encode_data = pd.DataFrame(en, columns=en_cols)
    X = pd.concat([x, encode_data], axis=1)

    # Handle imbalance
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Initialize KNN
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)

    # Evaluate
    y_pred = model.predict(x_test)

    st.subheader("Model Performance")
    st.write(f"**Train Score:** {model.score(x_train, y_train):.4f}")
    st.write(f"**Test Score:** {model.score(x_test, y_test):.4f}")
    st.write(f"**Precision:** {precision_score(y_test, y_pred):.4f}")
    st.write(f"**Recall:** {recall_score(y_test, y_pred):.4f}")
    st.write(f"**F1 Score:** {f1_score(y_test, y_pred):.4f}")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Save model and encoder for prediction tab
    st.session_state["model"] = model
    st.session_state["le"] = le
    st.session_state["en_cols"] = en_cols

# -------------------
# Predict New Patient Tab
# -------------------
elif page == "Predict New Patient":
    if "model" not in st.session_state:
        st.warning("Please train the model first in the 'Train Model' tab.")
    else:
        st.subheader("Predict Stroke for a New Patient")
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        hypertension = st.selectbox("Hypertension (0=No, 1=Yes)", [0,1])
        heart_disease = st.selectbox("Heart Disease (0=No, 1=Yes)", [0,1])
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        ever_married = st.selectbox("Ever Married", ["Yes","No"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Residence Type", ["Urban","Rural"])
        smoking_status = st.selectbox("Smoking Status", ["formerly smoked","never smoked","smokes","Unknown"])

        if st.button("Predict Stroke"):
            patient_df = pd.DataFrame({
                "age":[age],
                "hypertension":[hypertension],
                "heart_disease":[heart_disease],
                "avg_glucose_level":[avg_glucose_level],
                "bmi":[bmi],
                "gender":[gender],
                "ever_married":[ever_married],
                "work_type":[work_type],
                "Residence_type":[residence_type],
                "smoking_status":[smoking_status]
            })

            # Encode categorical features
            patient_en = st.session_state["le"].transform(patient_df[["gender","ever_married","work_type","Residence_type","smoking_status"]])
            patient_encoded = pd.DataFrame(patient_en, columns=st.session_state["en_cols"])
            patient_X = pd.concat([patient_df[["age","hypertension","heart_disease","avg_glucose_level","bmi"]], patient_encoded], axis=1)

            # Predict
            pred = st.session_state["model"].predict(patient_X)[0]
            st.success(f"The model predicts: **{'Stroke' if pred==1 else 'No Stroke'}**")
