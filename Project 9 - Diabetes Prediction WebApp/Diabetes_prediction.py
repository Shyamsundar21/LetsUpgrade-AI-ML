# Importing libraries

import pickle
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Dataset/diabetes.csv')
df.head()


df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness',
                                                                      'Insulin','BMI']].replace(0, np.NaN)

def median_target(var):   
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp

cols = df.columns
cols = cols.drop('Outcome')

for col in cols:
    median_target(col)
    
    df.loc[(df['Outcome'] == 0) & (df[col].isnull()), col] = median_target(col)[col][0]
    df.loc[(df['Outcome'] == 1) & (df[col].isnull()), col] = median_target(col)[col][1]
    
    
scaler = StandardScaler()
for col in df.columns[:-1]:
    df[col] = scaler.fit_transform(df[[col]])
    
    
X = df.drop('Outcome',axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)


model_final_random = RandomForestClassifier(n_estimators = 1000,
                                            min_samples_split = 11, 
                                            min_samples_leaf = 1, 
                                            max_features = 6)

model_final_random.fit(X_train, y_train)

y_pred_proba = model_final_random.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
print("ROC Score : ",roc_auc_score(y_test, y_pred_proba))
print("Accuracy for train: ", accuracy_score(y_train, model_final_random.predict(X_train)))
print("Accuracy for test: " , accuracy_score(y_test, model_final_random.predict(X_test)))

y_pred = model_final_random.predict(X_test)

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)

import pickle

with open('model_rf.pkl','wb') as file:
    pickle.dump(model_final_random,file)