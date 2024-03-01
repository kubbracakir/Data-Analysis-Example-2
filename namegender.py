import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/mac/Desktop/Python/src/Kübra/dataset.csv')
df.head()

df.describe()
df.info()
df.shape
df.isnull().sum()

sns.set(style = 'whitegrid')

num_gender = df["Gender"].value_counts()

plt.figure(figsize=(12,6))
sns.barplot(x = num_gender.index, y = num_gender.values)
plt.title('Gender Distribution')
plt.show()

plt.pie(num_gender, labels= num_gender.index, autopct = '%1.1f%%')
plt.title('Gender Distribution')
plt.axis('equal')
plt.show()

df = df.sort_values(by = ['Count'], ascending = False)
df.head()

def top_names(data, top=5):
    top_names = data.sort_values(by='Count', ascending=False).head(top)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_names, x='Name', y='Count', palette='viridis')
    plt.title(f'Top {top} Names by Count')
    plt.xlabel('Name')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

top_names(df)

import warnings
warnings.filterwarnings('ignore')

def name_length(data):
    df['Name_length'] = df['Name'].apply(len)
    plt.figure(figsize = (8,6))
    sns.histplot(df ['Name_length'], kde = True, color= 'skyblue')
    plt.title('Name Length Distribution')
    plt.xlabel('Name Length')
    plt.ylabel('Frequency')
    plt.show()
    
name_length(df)

from sklearn.preprocessing import LabelEncoder

X = df.drop(columns=['Gender'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Gender'])

categorical_cols = ['Name']
for col in categorical_cols:
    X[col] = label_encoder.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)
logistic_pred = logistic_model.predict(X_test_scaled)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)

def model_results(model,y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    cr= classification_report(y_true, y_pred)
    print(f'Model: {model}')
    print(accuracy)
    print(cr)

model_results(logistic_model, y_test, logistic_pred)

model_results("Random Forest", y_test, rf_pred)

model_results("Decision Tree", y_test, dt_pred)

from sklearn.metrics import roc_auc_score, roc_curve

logistic_auc = roc_auc_score(y_test, logistic_model.predict_proba(X_test_scaled)[:, 1])
logistic_fpr, logistic_tpr, _ = roc_curve(y_test, logistic_model.predict_proba(X_test_scaled)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(logistic_fpr, logistic_tpr, label=f'Logistic Regression (AUC = {logistic_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.show()

rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.show()

dt_auc = roc_auc_score(y_test, dt_model.predict_proba(X_test_scaled)[:, 1])
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_model.predict_proba(X_test_scaled)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {dt_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc='lower right')
plt.show()