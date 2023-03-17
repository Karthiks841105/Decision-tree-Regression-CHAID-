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
from student import CHAIDDecisionTreeRegressor
from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("std_classifier.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(school,sex,age,Medu,Fedu	,traveltime	,studytime	,failures	,schoolsup,famsup,paid,activities,nursery,internet,freetime,goout,absences,higher_yes,health_5,prev_grade):
    
    
   
    prediction=classifier.predict([[school,sex,age,Medu,Fedu	,traveltime	,studytime	,failures	,schoolsup,famsup,paid,activities,nursery,internet,freetime,goout,absences,higher_yes,health_5,prev_grade]])
    print(prediction)
    return prediction



def main():
    st.title("Student Performance Analysis")
    html_temp = """
    <body style="background-image: url("F:\Dockers-master\g1.jpg");
    background-size: cover;">
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Student Performance Analysis ML App </h2>
    </div>
    </body>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)

    display = ("Gabriel Pereira", "Mousinho da Silveira")
    options = list(range(len(display)))
    value1 = st.selectbox("school", options, format_func=lambda x: display[x])

    school= value1
    
   
    display = ("male", "female")
    options = list(range(len(display)))
    value = st.selectbox("gender", options, format_func=lambda x: display[x])
    sex=value
    
    age = st.text_input("Age","Type Here")

    display1 = ("NO","10th", "12th","Degree","Mastre Degree")
    options1 = list(range(len(display1)))
    value1 = st.selectbox("Mother_Education", options1, format_func=lambda x: display1[x])
    
    Medu = value1

    display2 = ("NO","10th", "12th","Degree","Mastre Degree")
    options2 = list(range(len(display2)))
    value2 = st.selectbox("Father_Education", options2, format_func=lambda x: display2[x])
    

    
    Fedu = value2
    
    traveltime = st.text_input("traveltime","Type Here")
    
    studytime = st.text_input("studytime","Type Here")
    
    failures = st.text_input("failures","Type Here")

    display3 = ("NO","Yes")
    options3= list(range(len(display3)))
    value3 = st.selectbox("schoolsup", options3, format_func=lambda x: display3[x])
    
    
    schoolsup = value3
    
    display4 = ("NO","Yes")
    options4= list(range(len(display4)))
    value4 = st.selectbox("Family_support", options4, format_func=lambda x: display4[x])
    famsup = value4
    

    display5 = ("NO","Yes")
    options5= list(range(len(display5)))
    value5 = st.selectbox("paid", options5, format_func=lambda x: display5[x])
    paid = value5
    
    activities = st.text_input("activities","Type Here")
    nursery = st.text_input("nursery","Type Here")
    internet = st.text_input("internet","Type Here")
    freetime = st.text_input("freetime","Type Here")
    goout = st.text_input("goout","Type Here")
    absences = st.text_input("absences","Type Here")
    higher_yes = st.text_input("higher_yes","Type Here")
    health_5 = st.text_input("health","Type Here")
    prev_grade = st.text_input("prev_grade","Type Here")
    
    
   
    result=""
    if st.button("Predict"):
        school=float(school)
        sex=float(sex)
        age=float(age)
        Medu=float(Medu)
        Fedu=float(Fedu)
        traveltime=float(traveltime)
        studytime=float(studytime)
        failures=float(failures)
        schoolsup=float(schoolsup)
        famsup=float(famsup)
        paid=float(paid)
        activities=float(activities)
        nursery=float(nursery)
        internet=float(internet)
        goout=float(goout)
        absences=float(absences)
        higher_yes=float(higher_yes)
        health_5=float(health_5)
        prev_grade=float(prev_grade)
        
        
        result=predict_note_authentication(school,sex,age,Medu,Fedu	,traveltime	,studytime	,failures	,schoolsup,famsup,paid,activities,nursery,internet,freetime,goout,absences,higher_yes,health_5,prev_grade)
    st.success('The Student Grade is  {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
        option = st.selectbox('How would you like to be contacted?',({'Email':1, 'Home phone':2, 'Mobile phone':3}))
        st.write('You selected:', option)

if __name__=='__main__':
    main()
    
    
    