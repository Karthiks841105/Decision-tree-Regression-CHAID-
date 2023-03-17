# -*- coding: utf-8 -*-
"""house_price.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tvTMgzmQYsc3PE5WvSsCyGB74Qz0-8CU
"""

# Importing all necessary Libaries

import numpy as np # numpy used for mathematical operation on array
import pandas as pd  # pandas used for data manipulation on dataframe
import seaborn as sns # seaborn used for data visualization
import matplotlib.pyplot as plt # matplotlib used for data visualization

df1=pd.read_csv("house(1).csv")
df1.head()

df2=pd.read_csv("house(2).csv")
df2.head()

df = pd.merge(df1, df2, on='id')

df = df.replace('?', np.nan)

df.head()

df.shape

df.nunique()

df.dtypes

df.info

df.describe(include='all')

# Plot histogram grid
df.hist(figsize=(16,16), xrot=-45) ## Display the labels rotated by 45 degress

# Clear the text "residue"
plt.show()

df.corr()

plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')

df.isna().sum()

df.nunique()

ca=a=df['bedrooms'].median()
df['bedrooms'].fillna(ca,inplace=True)
ca1=df['bathrooms'].median()
df['bathrooms'].fillna(ca1,inplace=True)

ca2=df['floors'].median()
df['floors'].fillna(ca2,inplace=True)

ca3=df['waterfront'].mode()[0]
df['waterfront'].fillna(ca3,inplace=True)

df.isna().sum()

df3=df.drop(['date','waterfront'],axis=1)

for k, v in df3.items():
        q1 = v.quantile(0.25)
        q3 = v.quantile(0.75)
        irq = q3 - q1
        v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
        perc = np.shape(v_col)[0] * 100.0 / np.shape(df3)[0]
        print("Column %s outliers = %.2f%%" % (k, perc))

for i in df3.columns:
  plt.boxplot(df3[i])
  plt.show()

'''for col in df3.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    df[col] = df[col].mask(df[col] < lower_bound, df[col].median(), axis=0)
    df[col] = df[col].mask(df[col] > upper_bound, df[col].median(), axis=0)
    df[col] = df[col].fillna(df[col].median())'''

df.date.head()

df['date'] = pd.to_datetime(df['date'])

df['year_sold'] = df['date'].dt.year

df['age'] = df['year_sold'] - df['yr_built']

df.drop(['id', 'date', 'yr_built','lat','long'], axis=1, inplace=True)

df.dtypes

df['waterfront'].unique()

df=pd.get_dummies(df, columns=['waterfront'],drop_first=True)

df.columns

df.dtypes

df.price.max()

df.nunique()

#writing  function  CHAIDDecisionTreeRegressor
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

def mean_squared_error1(y_true, y_pred):
   
      # Check if the lengths of both arrays are equal
      if len(y_true) != len(y_pred):
          raise ValueError("Length of y_true and y_pred should be the same.")
      
      # Calculate the squared differences between the true and predicted values
      squared_differences = [(y_true[i] - y_pred[i])**2 for i in range(len(y_true))]
      
      # Calculate the mean of the squared differences
      mse1 = sum(squared_differences) / len(squared_differences)
      
      return mse1

from sklearn.model_selection import train_test_split

X=df.drop('price',axis=1).values
y=df['price'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=CHAIDDecisionTreeRegressor()

model.fit(X_train,y_train)

ypred=model.predict(X_test)

print(ypred)

mean_squared_error1(y_test,ypred)

from sklearn.tree import DecisionTreeRegressor

model2=DecisionTreeRegressor()

model2.fit(X_train,y_train)

ypred=model2.predict(X_test)

print(ypred)

mean_squared_error1(ypred,y_test)

import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(model2, pickle_out)
pickle_out.close()



### Create a Pickle file using serialization 
