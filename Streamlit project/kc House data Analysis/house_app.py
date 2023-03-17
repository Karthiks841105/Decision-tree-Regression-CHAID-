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
from house_price import CHAIDDecisionTreeRegressor
from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(sqft_basement,yr_renovated,zipcode,sqft_living15,sqft_lot15,bedrooms,bathrooms,sqft_living,sqft_lot,floors,view,condition,grade,sqft_above,year_sold,age,waterfront_1):
    
    
   
    prediction=classifier.predict([[sqft_basement,yr_renovated,zipcode,sqft_living15,sqft_lot15,bedrooms,bathrooms,sqft_living,sqft_lot,floors,view,condition,grade,sqft_above,year_sold,age,waterfront_1]])
    print(prediction)
    return prediction



def main():
    st.title("House Sales Analysis")
    html_temp = """
    <body style="background-image: url("F:\Dockers-master\g1.jpg");
    background-size: cover;">
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">StreamlitHouse Sales Analysis ML App </h2>
    </div>
    </body>
    """
   
    st.markdown(html_temp,unsafe_allow_html=True)

    
    sqft_basement= st.text_input("sqft_basement","Type Here")
    
   
   
    yr_renovated=st.text_input("year_renovated","Type Here")
    
    zipcode = st.text_input("zipcode","Type Here")
    
    sqft_living15 = st.text_input("sqft_living15","Type Here")

    sqft_lot15 = st.text_input("sqft_lot15","Type Here")
    
    bedrooms = st.text_input("bedrooms","Type Here")
    
    bathrooms = st.text_input("bathrooms","Type Here")
    
    sqft_living = st.text_input("sqft_living","Type Here")

    sqft_lot = st.text_input("sqft_lot","Type Here")

    
    floors = st.text_input("floors","Type Here")
    view = st.text_input("view","Type Here")
    condition = st.text_input("condition","Type Here")
    grade = st.text_input("grade","Type Here")
    sqft_above = st.text_input("sqft_above","Type Here")
    year_sold = st.text_input("year_sold","Type Here")
    age = st.text_input("age","Type Here")
    waterfront_1 = st.text_input("waterfront_1","Type Here")
    
    result=""
    if st.button("Predict"):
        
        sqft_basement=float(sqft_basement)
        yr_renovated=float(yr_renovated)
        zipcode=float(zipcode)
        sqft_living15=float(sqft_living15)
        sqft_lot15=float(sqft_lot15)
        bedrooms=float(bedrooms)
        bathrooms=float(bathrooms)
        sqft_living=float(sqft_living)
        sqft_lot=float(sqft_lot)
        floors=float(floors)
        view=float(view)
        condition=float(condition)
        grade=float(grade)
        sqft_above=float(sqft_above)
        year_sold=float(year_sold)
        age=float(age)
        waterfront_1=float(waterfront_1)
        
        
        
        result=predict_note_authentication(sqft_basement,yr_renovated,zipcode,sqft_living15,sqft_lot15,bedrooms,bathrooms,sqft_living,sqft_lot,floors,view,condition,grade,sqft_above,year_sold,age,waterfront_1)
    st.success('House Sales Price is  {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
        option = st.selectbox('How would you like to be contacted?',({'Email':1, 'Home phone':2, 'Mobile phone':3}))
        st.write('You selected:', option)

if __name__=='__main__':
    main()
    
    
    