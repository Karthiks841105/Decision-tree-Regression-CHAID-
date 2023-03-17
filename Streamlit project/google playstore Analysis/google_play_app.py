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
from google_play import CHAIDDecisionTreeRegressor
from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(Reviews_x,Content_Rating,Genres,Last_Updated,	Current_Ver,Android_Ver,Category,Rating,Size,Installs,Type):
    
    
   
    prediction=classifier.predict([[Reviews_x,Content_Rating,Genres,Last_Updated,	Current_Ver,Android_Ver,Category,Rating,Size,Installs,Type]])
    print(prediction)
    return prediction



def main():
    st.title("Google Playstore  Analysis")
    html_temp = """
    <body style="background-image: url("F:\Dockers-master\g1.jpg");
    background-size: cover;">
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Google Playstore  ML App </h2>
    </div>
    </body>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)

    
    Reviews_x= st.text_input("Reviews","Type Here")
    
   
    display = ("Unrated","Everyone", "Teen","Everyone 10+","Mature 17+","Adults only 18+")
    options = list(range(len(display)))
    value = st.selectbox("Content_Rating", options, format_func=lambda x: display[x])
    Content_Rating=value
    
    Genres = st.text_input("Genres","Type Here")
    Last_Updated = st.text_input("Last_Updated","Type Here")

    Current_Ver = st.text_input("Current_Ver","Type Here")
    
    Android_Ver = st.text_input("Android_Ver","Type Here")
    
    Category = st.text_input("Category","Type Here")
    
    Rating = st.text_input("Rating","Type Here")

    Size = st.text_input("Size","Type Here")
    Installs = st.text_input("Installs","Type Here")
    Type = st.text_input("Type","Type Here")
    
    result=""
    if st.button("Predict"):
        Reviews_x=float(Reviews_x)
        Content_Rating=float(Content_Rating)
        Genres=float(Genres)
        Last_Updated=float(Last_Updated)
        Current_Ver=float(Current_Ver)
        Android_Ver=float(Android_Ver)
        Category=float(Category)
        Rating=float(Rating)
        Size=float(Size)
        Installs=float(Installs)
        Type=float(Type)
        
        
        
        result=predict_note_authentication(Reviews_x,Content_Rating,Genres,Last_Updated,Current_Ver,Android_Ver,Category,Rating,Size,Installs,Type)
    st.success('The App Price is  {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
        option = st.selectbox('How would you like to be contacted?',({'Email':1, 'Home phone':2, 'Mobile phone':3}))
        st.write('You selected:', option)

if __name__=='__main__':
    main()
    
    
    