#Import libraries
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, ConfusionMatrixDisplay

#Loading data
path='./Data/loan_data.csv'
df=pd.read_csv(path)

#Split test and train data
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

# Define features and target
X_train=df_train.drop(columns=['loan_status'])
y_train=df_train['loan_status']
X_test=df_test.drop(columns=['loan_status'])
y_test=df_test['loan_status']

#Building the pipeline
# Split the columns
num_cols = X_train.select_dtypes(include='number').columns.tolist()
cat_cols=X_train.select_dtypes(include='object').columns.tolist()
cat_ord_cols=[cat_cols[0],cat_cols[4]]
cat_hot_cols=[cat_cols[1],cat_cols[2],cat_cols[3]]

#Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols ),
        ('cat_ord',OrdinalEncoder(),cat_ord_cols),
        ('cat_hot',OneHotEncoder(sparse_output=False),cat_hot_cols)
    ]
)

#Define the Random Forest model
model= RandomForestClassifier(n_estimators=100,class_weight='balanced')

#Create the full pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', model)
])

#Fit the pipeline
full_pipeline.fit(X_train, y_train)

#Predict on the test set
y_pred = full_pipeline.predict_proba(X_test)

#Evaluate the model
accuracy=accuracy_score(y_test, np.argmax(y_pred,axis=1))
print(f"accuracy score is {accuracy}" )
ConfusionMatrixDisplay.from_predictions(y_test,np.argmax(y_pred,axis=1),normalize="true",values_format='.0%')
plt.show()

#Save the pipeline
joblib.dump(full_pipeline, 'pipeline_bank_loan.joblib')