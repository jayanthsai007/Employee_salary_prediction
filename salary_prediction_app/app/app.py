import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('models/best_model.pkl')

# Education to years mapping
edu_years = {
    'Preschool': 6, '1st-4th': 7, '5th-6th': 8, '7th-8th': 9,
    '9th': 10, '10th': 11, '11th': 12, '12th': 12,
    'HS-grad': 12, 'Some-college': 14, 'Assoc-acdm': 14,
    'Assoc-voc': 14, 'Bachelors': 16, 'Masters': 18,
    'Prof-school': 19, 'Doctorate': 20
}

# App title
st.title("Income Classification App (>50K or <=50K)")
st.markdown("Predict whether an individual's income exceeds $50K.")

# Sidebar input
type_option = st.sidebar.radio("Select Prediction Type", [
                               "Single Input", "Batch CSV"])

if type_option == "Single Input":
    age = st.slider("Age", 18, 90, 30)
    education = st.selectbox("Education", list(edu_years.keys()))
    occupation = st.selectbox("Occupation", [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
        'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical',
        'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
    ])
    hours = st.slider("Hours per week", 1, 99, 40)

    edu_yrs = edu_years[education]
    experience = max(0, age - edu_yrs - 6)

    input_dict = {
        'age': [age],
        'education': [education],
        'occupation': [occupation],
        'hours-per-week': [hours],
        'experience': [experience]
    }
    input_df = pd.DataFrame(input_dict)

    if st.button("Predict"):
        pred = model.predict(input_df)[0]
        label = ">50K" if pred == 1 else "<=50K"
        st.success(f"Predicted Income: {label}")

elif type_option == "Batch CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Compute experience if missing
        if 'experience' not in df.columns:
            df['edu_years'] = df['education'].map(edu_years)
            df['experience'] = df['age'] - df['edu_years'] - 6
            df['experience'] = df['experience'].apply(lambda x: max(0, x))

        preds = model.predict(df)
        df['predicted_income'] = np.where(preds == 1, '>50K', '<=50K')
        st.dataframe(df)

        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv,
                           "predictions.csv", "text/csv")
