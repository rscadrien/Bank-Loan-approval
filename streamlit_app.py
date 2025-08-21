import joblib
import pandas as pd
import streamlit as st

# Streamlit UI
st.title("Bank Loan Prediction (for small amounts of loan $0-50,000)")

#User Inputs
Age = st.number_input("What is the applicant's age?",min_value=18, max_value=100, step=1)
Gender =  st.selectbox("What is the applicant's gender?",["male","female"])
Education = st.selectbox("What is the applicant's education level?",["High school","Associate","Bachelor","Master","Doctorate"])
Income = st.number_input("What is the applicant's annual income?",min_value=0, step=100)
Experience = st.number_input("What is the applicant's years of experience?",min_value=0, max_value=50000, step=1)
Ownership = st.selectbox("Does the applicant own, rent, have a mortgage or other? ",["RENT","OWN","MORTGAGE","OTHER"])
Loan_Amount = st.number_input("What is the amount of loan the applicant want?",min_value=0, step=100)
Loan_Intent = st.selectbox("What is the applicant's loan intent?",['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
Int_rate = st.number_input("What is the interest rate?",min_value=0.0, max_value=100.0, step=0.1)
Credit_history = st.number_input("What is the length of the applicant's credit history in years?",min_value=0, step=1)
Credit_score = st.number_input("What is the applicant's credit score?",min_value=300, max_value=850, step=1)
Loan_default = st.selectbox("Does the applicant have a clean loan history (no default)? ",["No","Yes"])

if st.button("Predict Loan Approval Probability"):
    # Create a dataframe with the input
    X = pd.DataFrame([{
        'person_age' : Age,
        'person_gender' : Gender,
        'person_education' : Education,
        'person_income' : Income,
        'person_emp_exp' : Experience,
        'person_home_ownership' : Ownership,
        'loan_amnt' : Loan_Amount,
        'loan_intent' : Loan_Intent,
        'loan_int_rate' : Int_rate,
        'loan_percent_income' : Loan_Amount / Income,
        'cb_person_cred_hist_length' : Credit_history,
        'credit_score' : Credit_score,
        'previous_loan_defaults_on_file' : Loan_default
    }])
    # Load the model
    pipeline = joblib.load('pipeline_bank_loan.joblib')

    #Apply the model
    y_pred = pipeline.predict_proba(X)

    #Display the result
    st.write(f"The probability of the loan to be approved is : {y_pred[0][0]:.3f}") # Probability of approval is the first class
    st.write(f"The probability of the loan to be denied is : {y_pred[0][1]:.3f}")  # Probability of denial is the second class 