#!/usr/bin/env python
# coding: utf-8

# In[149]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# In[151]:


# Loading the dataset
df = pd.read_csv("heart.csv")
df


# In[153]:


# Checking the shape
df.shape


# In[155]:


df.head()


# In[157]:


# Checking for null 
df.isnull().sum()


# In[159]:


# One hot encoding the categorical variables
df_encoded = pd.get_dummies(
    df,
    columns=["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"],
    drop_first=True 
)


# In[161]:


# Setting the variables where X contains the features and y has the label
y = df_encoded['HeartDisease']
X = df_encoded.drop('HeartDisease', axis = 1)

#Number of features and examples 
print("Number of examples: " + str(X.shape[0]))
print("\nNumber of Features:" + str(X.shape[1]))
print(str(list(X.columns)))


# In[163]:


# Creating the train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=123)


# In[165]:


# Dimensions of test and training set
print(X_train.shape)
print(X_test.shape)


# In[167]:


# Training the Decision Tree
model = DecisionTreeClassifier(criterion = 'entropy', max_depth=4, random_state=123)

# Model fit
model.fit(X_train, y_train)

# Evaluating the Model 
dt_preds = model.predict(X_test)
dt_probs = model.predict_proba(X_test)[:, 1]

#Metrics
print("Accuracy:", accuracy_score(y_test, dt_preds ))
print("ROC-AUC:", roc_auc_score(y_test, dt_probs ))
print("\nClassification Report:\n", classification_report(y_test, dt_preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, dt_preds ))


# In[169]:


# Training Random Forest Model 
rf_model = RandomForestClassifier(criterion='entropy', n_estimators=100,max_depth=4, random_state=123)

# Model fit
rf_model.fit(X_train, y_train)

rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_preds = rf_model.predict(X_test)

#Evaluation Metrics
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("ROC-AUC:", roc_auc_score(y_test, rf_probs))
print("\nClassification Report:\n", classification_report(y_test, rf_preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, rf_preds))


# In[ ]:





# In[ ]:




