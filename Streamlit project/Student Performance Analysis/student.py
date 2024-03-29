# -*- coding: utf-8 -*-
"""student.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oBj0um5biibdIi4mhHsxqjAc2F_PLppy
"""

# Importing all necessary Libaries

import numpy as np # numpy used for mathematical operation on array
import pandas as pd  # pandas used for data manipulation on dataframe
import seaborn as sns # seaborn used for data visualization
import matplotlib.pyplot as plt # matplotlib used for data visualization

df1=pd.read_csv("students(1).csv")
df1.head()

df2=pd.read_csv("students(2).csv")
df2.head()

df = pd.merge(df1, df2, on='id')

df.shape

df1.shape

df2.shape

# Plot histogram grid
df2.hist(figsize=(16,16), xrot=-45) ## Display the labels rotated by 45 degress

# Clear the text "residue"
plt.show()

df2.corr()

plt.figure(figsize=(20,20))
sns.heatmap(df2.corr(),annot=True,cmap='coolwarm')

df.info

df.head()

df = df.replace('?', np.nan)

df.dtypes

df.nunique()

df.isna().sum()

d=df['famsize'].mode()[0]
df['famsize'].fillna(d,inplace=True)
e=df['reason'].mode()[0]
df['reason'].fillna(e,inplace=True)
f=df['higher'].mode()[0]
df['higher'].fillna(f,inplace=True)
g=df['health'].mode()[0]
df['health'].fillna(g,inplace=True)

df.isna().sum()

from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 

df['school']= label_encoder.fit_transform(df['school'])
df['sex']= label_encoder.fit_transform(df['sex'])

df['romantic']= label_encoder.fit_transform(df['romantic'])
df['internet']= label_encoder.fit_transform(df['internet'])

df['nursery']= label_encoder.fit_transform(df['nursery'])
df['activities']= label_encoder.fit_transform(df['activities'])
df['paid']= label_encoder.fit_transform(df['paid'])
df['famsup']= label_encoder.fit_transform(df['famsup'])
df['schoolsup']= label_encoder.fit_transform(df['schoolsup'])
df['guardian']= label_encoder.fit_transform(df['guardian'])

df['Fjob']= label_encoder.fit_transform(df['Fjob'])
df['Mjob']= label_encoder.fit_transform(df['Mjob'])
df['Pstatus']= label_encoder.fit_transform(df['Pstatus'])
df['famsize']= label_encoder.fit_transform(df['famsize'])
df['address']= label_encoder.fit_transform(df['address'])
df['sex']= label_encoder.fit_transform(df['sex'])

df.dtypes

df['higher'].unique()

df = pd.get_dummies(df, columns=['higher','health'],drop_first=True)

df.drop('reason',axis=1,inplace=True)

df.dtypes

df.drop('id',axis=1,inplace=True)

df.corr()

df.columns

for k, v in df.items():
        q1 = v.quantile(0.25)
        q3 = v.quantile(0.75)
        irq = q3 - q1
        v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
        perc = np.shape(v_col)[0] * 100.0 / np.shape(df)[0]
        print("Column %s outliers = %.2f%%" % (k, perc))

df['prev_grade'] = round(df['G1']+df['G2']/ 2)

df=df.drop(['G1','G2'],axis=1)

df.dtypes

df.drop('Fjob',axis=1,inplace=True)
df.drop('guardian',axis=1,inplace=True)
df.drop('romantic',axis=1,inplace=True)
df.drop('health_3',axis=1,inplace=True)

df.drop('Mjob',axis=1,inplace=True)
df.drop('address',axis=1,inplace=True)
df.drop('Pstatus',axis=1,inplace=True)
df.drop('health_4',axis=1,inplace=True)
df.drop('famsize',axis=1,inplace=True)
df.drop('famrel',axis=1,inplace=True)

df.shape

df.columns

df.drop('Dalc',axis=1,inplace=True)
df.drop('Walc',axis=1,inplace=True)
df.head()

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

X=df.drop('G3',axis=1).values
y=df['G3'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,)

model=CHAIDDecisionTreeRegressor()

model.fit(X_train,y_train)

ypred=model.predict(X_test)

df.columns

a=model.predict([[0.0,	0	,18.0	,4,	4.0,	2.0,	2.0,	0.0,	0.0,	0,	0,	1.0,	1.0,	3.0,	4,	6.0	,6	,1.0,1.0,	8.0]])
print(a)

print(ypred)

mean_squared_error1(y_test,ypred)

import pickle
pickle_out = open("std_classifier.pkl","wb")
pickle.dump(model, pickle_out)
pickle_out.close()

from sklearn.tree import DecisionTreeRegressor

model2=DecisionTreeRegressor()

model2.fit(X_train,y_train)

ypred=model2.predict(X_test)

print(ypred)

mean_squared_error1(ypred,y_test)

import pickle
pickle_out = open("classifier1.pkl","wb")
pickle.dump(model2, pickle_out)
pickle_out.close()

from collections import defaultdict
from random import randint
# Define the hyperparameters to be tuned
max_depth = [int(x) for x in np.linspace(2, 14, num = 11)]
min_samples_split = [2, 4, 6, 8, 10,12]
min_samples_leaf = [1, 2, 3, 4, 5,8,10]

# Define the number of iterations for hyperparameter tuning
n_iter = 20

# Define the number of folds for cross-validation
n_folds = None

# Define a function for k-fold cross-validation
def k_fold_cv(X, y, model, n_folds):
    # Initialize a dictionary to store the cross-validation scores
    scores = defaultdict(list)

    # Divide the data into k folds
    fold_size = len(X) // n_folds
    fold_starts = [i * fold_size for i in range(n_folds)]
    fold_ends = [(i + 1) * fold_size for i in range(n_folds)]
    fold_ends[-1] = len(X)

    # Perform k-fold cross-validation
    for i in range(n_folds):
        # Split the data into training and validation sets
        X_train = np.concatenate([X[:fold_starts[i]], X[fold_ends[i]:]])
        y_train = np.concatenate([y[:fold_starts[i]], y[fold_ends[i]:]])
        X_valid = X[fold_starts[i]:fold_ends[i]]
        y_valid = y[fold_starts[i]:fold_ends[i]]

        # Define the hyperparameters to be tuned
        params = {'max_depth': max_depth[randint(0, len(max_depth)-1)],
                  'min_samples_split': min_samples_split[randint(0, len(min_samples_split)-1)],
                  'min_samples_leaf': min_samples_leaf[randint(0, len(min_samples_leaf)-1)]}

        # Train the model with the current hyperparameters
        model.set_params(**params)
        model.fit(X_train, y_train)

        # Evaluate the model on the validation set
        y_pred = model.predict(X_valid)
        mse = mean_squared_error1(y_valid, y_pred)

        # Store the cross-validation score
        scores[mse].append(params)

    # Return the best hyperparameters and the corresponding mean squared error
    best_params = scores[min(scores)][0]
    best_mse = min(scores)

    return best_params, best_mse


# Define the decision tree regressor model
model1 = CHAIDDecisionTreeRegressor()

# Perform the hyperparameter tuning using k-fold cross-validation and randomized search
best_params, best_mse = k_fold_cv(X, y, model1, n_folds=10)

# Print the best hyperparameters and the corresponding mean squared error
print("Best hyperparameters:", best_params)
print("Best mean squared error:", best_mse)

