# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""


@author: karthik 
"""


import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 
from forest import CHAIDDecisionTreeRegressor
from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(X,Y,month,day,FFMC	,DMC	,DC	,ISI	,temp,RH,wind,rain):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[X,Y,month,day,FFMC	,DMC	,DC	,ISI	,temp,RH,wind,rain]])
    print(prediction)
    return prediction



def main():
    st.title("FOREST FIRES PREDICTION")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit FOREST FIRES PREDICTION ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    X = st.text_input("X","Type Here")
    
   
    Y = st.text_input("Y","Type Here")
    
    month = st.text_input("month","Type Here")
    
    day = st.text_input("day","Type Here")
    
    FFMC = st.text_input("FFMC","Type Here")
    
    DMC = st.text_input("DMC","Type Here")
    
    DC = st.text_input("DC","Type Here")
    
    ISI = st.text_input("ISI","Type Here")
    
    temp = st.text_input("temp","Type Here")
    
    RH = st.text_input("RH","Type Here")
    
    wind = st.text_input("wind","Type Here")
    
    rain = st.text_input("rain","Type Here")
   
    result=""
    if st.button("Predict"):
        X=float(X)
        Y=float(Y)
        month=float(month)
        day=float(day)
        FFMC=float(FFMC)
        DMC=float(DMC)
        DC=float(DC)
        ISI=float(ISI)
        temp=float(temp)
        RH=float(RH)
        wind=float(wind)
        rain=float(rain)
        
        
        result=predict_note_authentication(X,Y,month,day,FFMC	,DMC	,DC	,ISI	,temp,RH,wind,rain)
    st.success('The fired area is  {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    