# CS-4680-Homework-1

Heart Disease Prediction Dataset : https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data

**Target Variable:** 
<br>
Heart Disease
<br>
1: means there is a heart disease
<br>
0: means there is no heart disease

**Features:** 
1. Age
2. Chest Pain Typ
3. Cholesterol
4. Resting ECG
5. Max HR
6. Exercise Angina

# Results
In this assignment, I built a classification model to help predict whether a patient has heart disease or not based on clinical and demographic features. I used two models which are Random Forest and Decision Tree.

Based on the metrics, we can see that the Random Forest has outperformed the Decision Tree Model in terms of accuracy, ROC-AUC and the recall especially. Decision Tree has a lower recall with 0.87 compared to the 0.94 we got with Random Forest so it means that it is better at detecting the disease. There are also fewer false negatives with Random Forest which means that Decison Tree is more risky because in terms of healthcare, it is important that we do not miss those who have the heart disease.

