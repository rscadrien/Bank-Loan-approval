import joblib
import pandas as pd
from Inference.valid_input import get_valid_input_str, get_valid_input_int, get_valid_input_float

#Ask for Input
Age = get_valid_input_float("What is the applicant's age? ",18,100)
Gender = get_valid_input_str("What is the applicant's gender? ",["female","male"])
Education = get_valid_input_str("What is the applicant's education level? ",["High school","Associate","Bachelor","Master","Doctorate"])
Income = get_valid_input_float("What is the applicant's annual income? ",0,1000000000)
Experience = get_valid_input_int("How many years of experience do the applicant have? ",0,100)
Ownership = get_valid_input_str("Does the applicant own, rent, have a mortgage or other? ",["RENT","OWN","MORTGAGE","OTHER"])
Loan_Amount = get_valid_input_float("What is the amount of loan the applicant want? ",0,1000000000)
Loan_Intent = get_valid_input_str("What is the applicant's loan intent? ",['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
Int_rate = get_valid_input_float("What is the interest rate? ",0,100)
Credit_history = get_valid_input_float("What is the length of the applicant's credit history in years? ",0,50)
Credit_score = get_valid_input_int("What is the applicant's credit score? ",300,850)
Loan_default = get_valid_input_str("Does the applicant have any loan default? ",["No","Yes"])

#Create a dataframe with the input
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

#Load the model
pipeline = joblib.load('pipeline_bank_loan.joblib')

#Apply the model
y_pred = pipeline.predict_proba(X)

#Printing the result
print(f"The probability of the loan to be approved is: {y_pred[0][0]:.3f}")