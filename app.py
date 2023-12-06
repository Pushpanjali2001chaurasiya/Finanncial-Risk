import pandas as pd
import numpy as np
import streamlit as st

st.title('Financial Risk Prediction')
st.markdown('This app predicts the :blue[financial risk] of a person based on its financial statements.')

st.header('User Input Parameters')
col1, col2, col3 = st.columns(3)
with col1:
   city = st.number_input('City Code from [0-44]', min_value=0, max_value=44, value=1)
   Location_Score = st.number_input('Location_Score from [0-100]', min_value=0.00, max_value=100.00, value=10.00)
with col2:
   Internal_Audit_Score = st.number_input('Internal_Audit_Score from [0-15]', min_value=0, max_value=15, value=1)
   External_Audit_Score = st.number_input('External_Audit_Score from [0-15]', min_value=0, max_value=15, value=1)
with col3:
   Fin_Score = st.number_input('Fin_Score from [0-15]', min_value=0, max_value=15, value=1)
   Loss_score = st.number_input('Loss_score from [0-13]', min_value=0, max_value=13, value=1)
   Past_Results = st.slider('Past_Results', 0,10,1)

if st.button('Risk Prediction'):
   # Read the dataset
   df = pd.read_csv('Train.csv')
   df = df.drop(['IsUnderRisk'],axis=1)
   # Load the model
   import pickle
   model = pickle.load(open('logistic_model.pkl', 'rb'))
   # Apply model to make predictions
   prediction = model.predict([[city,Location_Score,Internal_Audit_Score,External_Audit_Score,Fin_Score,Loss_score,Past_Results]])
   if prediction == 0:
      st.markdown('The :blue[financial risk] of this person is :green[Low]')
   else:
      st.markdown('The :blue[financial risk] of this person is :red[High]')
