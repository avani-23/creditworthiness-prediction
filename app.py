
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Creditworthiness Predictor")

st.title("💳 Creditworthiness Prediction App")
st.markdown("This app predicts whether an individual is likely to be creditworthy based on financial attributes.")

with open('credit_model.pkl', 'rb') as file:
    model = pickle.load(file)


def user_input():
    status = st.selectbox("Account Status", [0, 1, 2, 3])
    duration = st.slider("Duration (in months)", 4, 72)
    credit_history = st.selectbox("Credit History", [0, 1, 2, 3, 4])
    purpose = st.selectbox("Purpose", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    credit_amount = st.number_input("Credit Amount", 0, 20000)
    savings = st.selectbox("Savings", [0, 1, 2, 3, 4])
    employment = st.selectbox("Employment (Years)", [0, 1, 2, 3, 4])
    installment_rate = st.slider("Installment Rate", 1, 4)
    personal_status = st.selectbox("Personal Status", [0, 1, 2, 3])
    other_debtors = st.selectbox("Other Debtors", [0, 1, 2])
    residence = st.slider("Years at Residence", 1, 4)
    property = st.selectbox("Property", [0, 1, 2, 3])
    age = st.slider("Age", 18, 75)
    other_plans = st.selectbox("Other Installment Plans", [0, 1, 2])
    housing = st.selectbox("Housing", [0, 1, 2])
    credits = st.slider("Number of Credits", 1, 4)
    job = st.selectbox("Job Type", [0, 1, 2, 3])
    liable = st.slider("People Liable", 1, 2)
    telephone = st.selectbox("Telephone", [0, 1])
    foreign_worker = st.selectbox("Foreign Worker", [0, 1])

    data = {
        'Status': status, 'Duration': duration, 'CreditHistory': credit_history, 'Purpose': purpose,
        'CreditAmount': credit_amount, 'Savings': savings, 'Employment': employment,
        'InstallmentRate': installment_rate, 'PersonalStatus': personal_status, 'OtherDebtors': other_debtors,
        'ResidenceSince': residence, 'Property': property, 'Age': age, 'OtherInstallmentPlans': other_plans,
        'Housing': housing, 'NumberCredits': credits, 'Job': job, 'PeopleLiable': liable,
        'Telephone': telephone, 'ForeignWorker': foreign_worker
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = " Creditworthy" if prediction == 1 else " Not Creditworthy"
    st.subheader(f"Prediction: {result}")
