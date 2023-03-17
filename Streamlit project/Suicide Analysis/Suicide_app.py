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
from suicide_prevention import CHAIDDecisionTreeRegressor
from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(HDI_for_year,gdp_for_year,gdp_per_capita,generation,country,year,sex,age,population,suicides_pop):
    
    
   
    prediction=classifier.predict([[HDI_for_year,gdp_for_year,gdp_per_capita,generation,country,year,sex,age,population,suicides_pop]])
    print(prediction)
    return prediction



def main():
    st.title("Suicide Prevention  Analysis")
    html_temp = """
    <body style="background-image: url("F:\Dockers-master\g1.jpg");
    background-size: cover;">
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Suicide_Prevention  ML App </h2>
    </div>
    </body>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)

    
    HDI_for_year= st.text_input("HDI_for_year","Type Here")
    
   
    
    
    gdp_for_year = st.text_input("gdp_for_year","Type Here")
    gdp_per_capita = st.text_input("gdp_per_capita","Type Here")

    generation = st.text_input("generation","Type Here")
    
    country = st.text_input("country","Type Here")
    
    year = st.text_input("year","Type Here")
    
    display = ("Male","Female")
    options = list(range(len(display)))
    value = st.selectbox("Gender", options, format_func=lambda x: display[x])
    sex=value
    display1 = ('75+ years', '15-24 years', '25-34 years', '35-54 years',
       '5-14 years', '55-74 years')
    options1 = list(range(len(display1)))
    value1 = st.selectbox("Age", options1, format_func=lambda x: display1[x])
    age = value1
    population = st.text_input("population","Type Here")
    suicides_pop = st.text_input("suicides_pop","Type Here")
    
    result=""
    if st.button("Predict"):
        
        HDI_for_year=float(HDI_for_year)
        gdp_for_year=float(gdp_for_year)
        gdp_per_capita=float(gdp_per_capita)
        generation=float(generation)
        country=float(country)
        year=float(year)
        sex=float(sex)
        age=float(age)
        population=float(population)
        suicides_pop=float(suicides_pop)
        
        
        
        result=predict_note_authentication(HDI_for_year,gdp_for_year,gdp_per_capita,generation,country,year,sex,age,population,suicides_pop)
    st.success('Number of  Suicide  is  {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
        option = st.selectbox('How would you like to be contacted?',({'Email':1, 'Home phone':2, 'Mobile phone':3}))
        st.write('You selected:', option)

if __name__=='__main__':
    main()
    
    
    