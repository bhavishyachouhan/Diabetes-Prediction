# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:13:17 2024

@author: HP
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import streamlit as st


loaded_model=pickle.load(open(r"C:/Users\HP/Downloads/ml project/trained_model (1).sav",'rb'))


#ceating a function for prediction
def diabetes_prediction(input_data):
    
#changing the input_data to numpy array
    input_data_as_numpy_array=np.asarray(input_data)
#reshaping the array as we are predicting for only one instance or record
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    scaler =StandardScaler()
    input_standardized=scaler.fit_transform(input_data_reshaped)
    prediction=loaded_model.predict(input_standardized)
    if(prediction[0]==0):
       return 'person is non diabetic'
    else:
       return 'person is diabetic'
   
def main():
    st.title('Diabetes prediction system')
    
    #getting input data from the user
    
    Pregnancies=st.text_input('numbers of Pregnancies')
    Glucose=st.text_input(' Glucose level')
    BloodPressure=st.text_input('BloodPressure value')
    SkinThickness=st.text_input('Skin thickness value')
    Insulin=st.text_input('Insulin')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function ')
    Age=st.text_input('age of the person')
    
    Pregnancies = int(Pregnancies)
    Glucose = int(Glucose)
    BloodPressure = int(BloodPressure)
    SkinThickness = int(SkinThickness)
    Insulin = int(Insulin)
    BMI = float(BMI)  # BMI might have decimal values, so float is appropriate
    DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
    Age = int(Age)
     #code for prediction
    diagnosis=''
     #creating button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
    st.success(diagnosis)   
        
if __name__=='__main__':
   main()        