
# Importing Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor

from sklearn import preprocessing 
from scipy import stats
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# label_encoder object knows how to understand word labels. 

class CHAIDDecisionTreeRegressor:
    
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth  #determines the maximum depth of the decision tree that will be constructed
        self.min_samples_split = min_samples_split #specifies the minimum number of samples required to split an internal node
        self.tree = {}
        
   
    def mse(self, y):
        # Calculate mean squared error of targets
        return np.mean((y - np.mean(y))**2)
    
    
   
    def split_data(self, feature, split, X, y):
        left_indices = np.where(X[:, feature] < split) # find the indices of samples where the feature is less than the split value,(np.where)
        right_indices = np.where(X[:, feature] >= split) # find the indices of samples where the feature is greater than or equal to the split value
        X_left = X[left_indices] #create a new array of input samples for the left node using the left indices
        y_left = y[left_indices] #create a new array of labels for the left node using the left indice
        X_right = X[right_indices] # create a new array of input samples for the right node using the right indices
        y_right = y[right_indices] # create a new array of labels for the right node using the right indices
        return X_left, y_left, X_right, y_right # return the new input and label arrays for the left and right nodes
   
   
    def chi_squared_test(self, x, y):
        # Perform chi-squared test to determine if a split is significant
        n_total = len(y) # calculate the total number of samples
        n_left = len(np.where(x < x.mean())[0]) # calculate the number of samples in the left node (where x is less than the mean of x)
        n_right = n_total - n_left # calculate the number of samples in the right node
        p_left = n_left / n_total # calculate the   samples in the left node
        p_right = n_right / n_total # calculate the  samples in the right node
        y_left_mean = np.mean(y[x < x.mean()]) # calculate the mean of the labels for the samples in the left node
        y_right_mean = np.mean(y[x >= x.mean()])  # calculate the mean of the labels for the samples in the right node
        y_total_mean = np.mean(y) # calculate the mean of the labels for all the samples
        chi_squared = (n_left * (y_left_mean - y_total_mean)**2 / (p_left * (1 - p_left)) +    # calculate the first part of the chi-squared test statistic
                       n_right * (y_right_mean - y_total_mean)**2 / (p_right * (1 - p_right)))  # calculate the second part of the chi-squared test statistic
        return chi_squared 
   
    def find_split(self, X, y):
        best_feature, best_split, best_chi2 = None, None, 0
        n_features = X.shape[1] #This line finds the number of features in the feature matrix X
        for feature in range(n_features): #sets up a loop that iterates over each feature in the input feature matrix X
            for split in np.unique(X[:, feature]): #sets up a nested loop that iterates over the unique values of the feature 
                chi2 = self.chi_squared_test(X[:, feature], y) #passing in the values of the feature and target variable for the current split
                if chi2 > best_chi2: 
                    best_feature, best_split, best_chi2 = feature, split, chi2 # chi-squared test statistic are stored in the best_feature, best_split, and best_chi2 variables.
        return best_feature, best_split
   
   
    def build_tree(self, X, y, depth):
        # Recursively build the decision tree
        n_samples, n_features = X.shape
        # Check for stopping criteria
        if depth == self.max_depth or n_samples < 2*self.min_samples_split:
            leaf_value = np.mean(y)
            return leaf_value
        best_feature, best_split = self.find_split(X, y) #index of the best feature to split on and the best threshold to use for that feature
        if best_feature is None:
            leaf_value = np.mean(y)
            return leaf_value
        #The method returns four NumPy arrays: X_left and y_left, which contain the subset of the dataset
        # where the feature value is less than or equal to the threshold, and X_right and y_right,
        # which contain the subset of the dataset where the feature value is greater than the threshold
        X_left, y_left, X_right, y_right = self.split_data(best_feature, best_split, X, y)
        #This line is creating a Python dictionary called decision_node that represents a decision node in the decision tree
        decision_node = {"feature": best_feature, "split": best_split, "left": None, "right": None}
        decision_node["left"] = self.build_tree(X_left, y_left, depth+1)# The value is the result of a recursive call to the build_tre
        decision_node["right"] = self.build_tree(X_right, y_right, depth+1)
        return decision_node
    
    
    def fit(self, X, y):
          # Build the decision tree
          self.tree = self.build_tree(X, y, depth=0)
    
    
    def predict_single(self, x):
        # Traverse the decision tree to make a prediction for a single instance
        node = self.tree
        while isinstance(node, dict):
            if x[node["feature"]] < node["split"]:
                node = node["left"]
            else:
                node = node["right"]
        return node
    
    def predict(self, X):
        # Make predictions for multiple instances
        return np.array([self.predict_single(x) for x in X])
    
    # writing  function for r2_score (R2 = 1 - (SSres / SStot)) 
    def r2(self,y_true, y_pred):
      # Calculate the mean of the true values
      y_true_mean = sum(y_true) / len(y_true)
      
      # Calculate the total sum of squares (TSS)
      tss = sum((y_true - y_true_mean) ** 2)
      
      # Calculate the residual sum of squares (RSS)
      rss = sum((y_true - y_pred) ** 2)
      
      # Calculate the R-squared value
      r2_score = 1 - (rss / tss)
      
      return r2_score


    # To calaulate the MSE 
    def score(self,y_true, y_pred):
   
      # Check if the lengths of both arrays are equal
      if len(y_true) != len(y_pred):
          raise ValueError("Length of y_true and y_pred should be the same.")
      
      # Calculate the squared differences between the true and predicted values
      squared_differences = [(y_true[i] - y_pred[i])**2 for i in range(len(y_true))]
      
      # Calculate the mean of the squared differences
      mse1 = sum(squared_differences) / len(squared_differences)
      
      return mse1
    
    def set_params(self, **kwargs):#**kwargs allows us to pass a variable number of keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self





label_encoder = preprocessing.LabelEncoder() 
data=''
# Setting up the header 
st.title("Dataset")
st.subheader("Complete Model Lifecycle")


Choose_file  = st.selectbox("Select filfe upload type", ("Single_file", "Two_file",))

if Choose_file== "Single_file":
    filename = st.file_uploader("upload file", type = ("csv", "xlsx"))
    data = pd.read_csv(filename,na_values=['?', '/', '#',''])
elif Choose_file =='Two_file':

    # Upload the first dataset
    df1 = st.file_uploader("Upload the first dataset", type=["csv", "xlsx"])

    # Upload the second dataset
    df2 = st.file_uploader("Upload the second dataset", type=["csv", "xlsx"])

    # Merge the two datasets
    if df1 is not None and df2 is not None:
        df1 = pd.read_csv(df1,na_values=['?', '/', '#','']) # Use pd.read_excel(df1) for Excel files
        df2 = pd.read_csv(df2,na_values=['?', '/', '#','']) # Use pd.read_excel(df2) for Excel files
        data = pd.merge(df1, df2, on='id')
        st.write(data)
    else:
        st.write("Please upload both datasets.")


# Providing a radio button to browse and upload the imput file 


#------------------------------------------------------------------------------
# To upload an input file from the specified path
#@st.cache(persist=True)
#def explore_data(dataset):
#    df = pd.read_csv(os.path.join(dataset))
#    return df
#data = explore_data(my_dataset)
#------------------------------------------------------------------------------
def mean_squared_error1(y_true, y_pred):
   
      # Check if the lengths of both arrays are equal
      if len(y_true) != len(y_pred):
          raise ValueError("Length of y_true and y_pred should be the same.")
      
      # Calculate the squared differences between the true and predicted values
      squared_differences = [(y_true[i] - y_pred[i])**2 for i in range(len(y_true))]
      
      # Calculate the mean of the squared differences
      mse1 = sum(squared_differences) / len(squared_differences)
      print(mse1)
      return mse1

from sklearn.metrics import r2_score

def r2(y_true, y_pred):
    # Calculate the mean of the true values
    y_true_mean = sum(y_true) / len(y_true)

    # Calculate the total sum of squares (TSS)
    tss = sum((y_true - y_true_mean) ** 2)

    # Calculate the residual sum of squares (RSS)
    rss = sum((y_true - y_pred) ** 2)

    # Calculate the R-squared value
    r2_score = 1-(rss / tss)

    return r2_score

def remove_outliers(data):
    z_scores = np.abs(stats.zscore(data))
    data_clean = data[(z_scores < 3).all(axis=1)]
    return data_clean


def fill_outliers(data, method='zscore', axis=0):
    if method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        threshold = 3
        data[z_scores > threshold] = np.nan
        data.fillna(data.median(), inplace=True)
    elif method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        data[(data < lower_bound) | (data > upper_bound)] = np.nan
        data.fillna(data.median(), inplace=True)

    if axis == 0:
        return data
    elif axis == 1:
        return data.T

def drop_outliers(data, method='zscore', axis=0):
    if method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        threshold = 3
        data[z_scores > threshold] = np.nan
        data.dropna(axis=axis, inplace=True)
    elif method == 'iqr':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        data[(data < lower_bound) | (data > upper_bound)] = np.nan
        data.dropna(axis=axis, inplace=True)

    return data

# Dataset preview
if st.checkbox("Preview Dataset"):
    if st.button("Head"):
        st.write(data.head())
    elif st.button("Tail"):
        st.write(data.tail())
    else:
        number = st.slider("Select No of Rows", 1, data.shape[0])
        st.write(data.head(number))


# show entire data
if st.checkbox("Show all data"):
    st.write(data)

st.subheader('To Check Columns Name')
# show column names
if st.checkbox("Show Column Names"):
    st.write(data.columns)

# show dimensions
if st.checkbox("Show Dimensions"):
    st.write(data.shape)

st.subheader('Summaery of the Data')     
# show summary
if st.checkbox("Show Summary"):
    st.write(data.describe())

numeric_columns = data.select_dtypes(include=['int', 'float'])
st.subheader('Check null values and fill null values ')   
# show missing values
if st.checkbox("Show Missing Values"):
    st.write(numeric_columns.isna().sum())    

# Select a column to treat missing values
col_option = st.multiselect("Select Feature to fillna",numeric_columns.columns)

# Specify options to treat missing values
missing_values_clear = st.selectbox("Select Missing values treatment method", ("Replace with Mean", "Replace with Median", "Replace with Mode"))

if missing_values_clear == "Replace with Mean":
    replaced_value = data[col_option].mean()
    data[col_option]=data[col_option].mean()
    st.write("Mean value of column is :", replaced_value)
elif missing_values_clear == "Replace with Median":
    replaced_value = data[col_option].median()
    st.write("Median value of column is :", replaced_value)
elif missing_values_clear == "Replace with Mode":
    replaced_value = data[col_option].mode()
    st.write("Mode value of column is :", replaced_value)


Replace = st.selectbox("Replace values of column?", ("Yes", "No"))
if Replace == "Yes":
    data[col_option] = data[col_option].fillna(replaced_value,)
    st.write("Null values replaced")
elif Replace == "No":
    st.write("No changes made")

st.subheader(' Check Null values Categorical Columns and fill Null values  ')
#only categorical columns
object_columns = data.select_dtypes(include=['object'])
if st.checkbox("Show Missing Values of object columns"):
    st.write(object_columns.isna().sum()) 

col_option1 = st.multiselect("Select Feature to fillna",object_columns.columns)


# Specify options to treat missing values
missing_values_clear = st.selectbox("Select Missing values For Categorycal columns treatment method", ("Replace with Mean", "Replace with Median", "Replace with Mode"))

if missing_values_clear == "Replace with Mean":
    replaced_value1 = data[col_option1].mean()
    
    st.write("Mean value of column is :", replaced_value1)
elif missing_values_clear == "Replace with Median":
    replaced_value1 = data[col_option1].median()
    st.write("Median value of column is :", replaced_value1)
elif missing_values_clear == "Replace with Mode":
    replaced_value1 ='Missinig'
    
    st.write("Mode value of column is :", replaced_value1)



Replace = st.selectbox("Replace values of column to category?", ("Yes", "No"))
if Replace == "Yes":
    data[col_option1] = data[col_option1].fillna(replaced_value1,)
    st.write("Null values replaced")
elif Replace == "No":
    st.write("No changes made")



if st.checkbox("Show Missing   Values after fill"):
    st.write(data.isna().sum()) 
# To change datatype of a column in a dataframe
# display datatypes of all columns
if st.checkbox("Show datatypes of the columns"):
    st.write(data.dtypes)

st.subheader('Convert Datatype')
col_option_datatype = st.multiselect("Select Column to change datatype", data.columns) 

input_data_type = st.selectbox("Select Datatype of input column", (str,int, float))  
output_data_type = st.selectbox("Select Datatype of output column", (label_encoder,'OneHot_encode'))

st.write("Datatype of ",col_option_datatype," changed to ", output_data_type)
if output_data_type=='OneHot_encode':
    for i in col_option_datatype:
        data = pd.get_dummies(data, columns=[i],drop_first=True)
        
else:
    for i in col_option_datatype:
        data[i] = output_data_type.fit_transform(data[i])
        


if st.checkbox("Show updated datatypes of the columns"):
    st.write(data.dtypes)

if st.checkbox("Preview Dataset aftre convert datatype"):
    if st.button("Head "):
        st.write(data.head())

st.subheader(' Check Outliers and Replace Outliers')
show_outliers = st.checkbox("Show outliers")

# Display data with or without outliers
if show_outliers:
    for k, v in data.items():
            q1 = v.quantile(0.25)
            q3 = v.quantile(0.75)
            irq = q3 - q1
            v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
            perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
            print("Column %s outliers = %.2f%%" % (k, perc))
            st.write(k,perc)


method = st.selectbox("Select outlier detection method", ("IQR", "Z-score"))

if st.checkbox("Fill Outliers"):
    if method == "IQR":
        data = fill_outliers(data, method='iqr', axis=0)
    elif method == "Z-score":
        data = fill_outliers(data, method='zscore', axis=0)

    st.write("Data with filled outliers")
    st.write(data)

if st.checkbox("Drop Outliers"):
    if method == "IQR":
        data = drop_outliers(data, method='iqr', axis=0)
    elif method == "Z-score":
        data = drop_outliers(data, method='zscore', axis=0)

    st.write("Data with dropped outliers")
    st.write(data)




show_outliers = st.checkbox("Show outliers aftre treatement")

# Display data with or without outliers
if show_outliers:
    for k, v in data.items():
            q1 = v.quantile(0.25)
            q3 = v.quantile(0.75)
            irq = q3 - q1
            v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
            perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
            print("Column %s outliers = %.2f%%" % (k, perc))
            st.write(k,perc)
# visualization
st.subheader('Scatter Plot')
# scatter plot
col1 = st.selectbox('Which feature on x?', data.columns)
col2 = st.selectbox('Which feature on y?', data.columns)
fig = px.scatter(data, x =col1,y=col2)
st.plotly_chart(fig)

st.subheader('Correlation Plot') 
# correlartion plots
if st.checkbox("Show Correlation plots with Seaborn"):
    st.write(sns.heatmap(data.corr()))
    st.pyplot()

st.subheader('Feature_Scaling')
scaling_method = st.selectbox('Select a scaling method:', ['Standardization', 'Normalization'])

# Perform the selected scaling method on the dataset
if scaling_method == 'Standardization':
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
else:
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

# Display the scaled data
st.write('Scaled data:')
st.write(pd.DataFrame(scaled_data, columns=data.columns))

# Machine Learning Algorithms
st.subheader('Machine Learning models')
 
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
 
 
features = st.multiselect("Select Feature Columns",data.columns)
labels = st.multiselect("select target column",data.columns)

features= data[features].values
labels = data[labels].values


train_percent = st.slider("Select % to train model", 1, 100)
train_percent = train_percent/100

X_train,X_test, y_train, y_test = train_test_split(features, labels, train_size=train_percent, random_state=1)


alg = ['XGBoost Classifier', 'Support Vector Machine', 'Random Forest Classifier','DecisionTreeRegressor','CHAIDDecisionTreeRegressor']
classifier = st.selectbox('Which algorithm?', alg)
if classifier=='XGBoost Classifier':
    XG = XGBClassifier()
    XG.fit(X_train, y_train)
    acc = XG.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_XG = XG.predict(X_test)
    cm_XG=confusion_matrix(y_test,pred_XG)
    st.write('Confusion matrix: ', cm_XG)
   
elif classifier == 'Support Vector Machine':
    svm=SVC()
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_svm = svm.predict(X_test)
    cm=confusion_matrix(y_test,pred_svm)
    st.write('Confusion matrix: ', cm)

elif classifier == 'Random Forest Classifier':
    RFC=RandomForestClassifier()
    RFC.fit(X_train, y_train)
    acc = RFC.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_RFC = RFC.predict(X_test)
    cm=confusion_matrix(y_test,pred_RFC)
    st.write('Confusion matrix: ', cm)

elif classifier == 'DecisionTreeRegressor':
    RFC=DecisionTreeRegressor()
    RFC.fit(X_train, y_train)
    #acc = RFC.score(X_test, y_test)
    #st.write('Accuracy: ', acc)
    pred_RFC = RFC.predict(X_test)
    cm=mean_squared_error1(y_test,pred_RFC)
    r2s=r2_score(y_test,pred_RFC)
    st.write('Mean_Squared_Error: ', cm)
    st.write('R2_Score: ', r2s)

elif classifier == 'CHAIDDecisionTreeRegressor':
    RFC=CHAIDDecisionTreeRegressor()
    RFC.fit(X_train, y_train)
    #acc = RFC.score(X_test, y_test)
    #st.write('Accuracy: ', acc)
    pred_RFC = RFC.predict(X_test)
    cm=mean_squared_error1(y_test,pred_RFC)
    print(cm)
    r2s=r2_score(y_test,pred_RFC)
    st.write('Mean_Squared_Error: ', cm)
    st.write('R2_Score: ', r2s)