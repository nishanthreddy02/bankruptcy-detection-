#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Streamlit app
st.title('Bankruptcy Prevention Prediction')
st.sidebar.header('User Input Parameters')
st.write('Enter the company details to predict bankruptcy risk:')

def user_input_features():
    # Create input fields
    industrial_risk = st.sidebar.selectbox('Industrial Risk',[1.0, 0.5, 0.0])
    management_risk = st.sidebar.selectbox('Management Risk',[1.0, 0.5, 0.0])
    financial_flexibility = st.sidebar.selectbox('Financial Flexibility',[1.0, 0.5, 0.0])
    credibility = st.sidebar.selectbox('Credibility',[1.0, 0.5, 0.0])
    competitiveness = st.sidebar.selectbox('Competitiveness',[1.0, 0.5, 0.0])
    operating_risk = st.sidebar.selectbox('Operating Risk',[1.0, 0.5, 0.0])
    
    data = {
        'industrial_risk': industrial_risk,
        'management_risk': management_risk,
        'financial_flexibility': financial_flexibility,
        'credibility': credibility,
        'competitiveness': competitiveness,
        'operating_risk': operating_risk
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader('User Input Parameters')
st.write(df)

# Load the CSV file
df_full = pd.read_csv(r"C:\Users\Microsoft\Downloads\bank123@.csv")

# Display the loaded data
st.write(df_full)

# Preprocess the data and split into features (X) and target (y)
X = df_full.drop('class', axis=1) 
y = df_full['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save the trained model to a file
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the saved model
loaded_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

# Scale the user input data
user_input_scaled = scaler.transform(df)
# Make predictions using the loaded model
prediction = loaded_model.predict(user_input_scaled)
prediction_probs = loaded_model.predict_proba(user_input_scaled)

# Display the predicted result
st.subheader('Predicted result')
if prediction[0] == 1:
    st.write('Sample: Yes (Bankruptcy risk)')
    st.write('Probability of Bankruptcy: {:.2f}%'.format(prediction_probs[0][1]*100))
else:
    st.write('Sample: No (No bankruptcy risk)')
    st.write('Probability of Non-Bankruptcy: {:.2f}%'.format(prediction_probs[0][0]*100))

st.subheader('Prediction Probability')
st.write(prediction_probs)


# In[ ]:





# In[ ]:




