import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle

#Ignore warnings
warnings.filterwarnings('ignore')

#Reading the dataset
data = pd.read_csv(r"C:\Users\hp\Downloads\DiabetesDetection\DiabetesDetection\diabetesDataset.csv")
# #Check missing values
data.isnull().sum()

#Splitting the dataset into dependent and independent
X = data.drop(['Outcome'], axis=1)
y = data['Outcome']

#split into traing and validating dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=2)

#Builidng the model using Logistic Regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

from sklearn import metrics
print("Accuracy_Score =", format(metrics.accuracy_score(y_test, lr_pred)))

# Building the model using KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)

from sklearn import metrics
print("Accuracy_Score =", format(metrics.accuracy_score(y_test,knn_pred)))


#Building model using Naive Byes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
gnb_pred = gnb.predict(X_test)

#Getting accuracy score for Naive Byes
from sklearn import metrics
print("Accuracy_Score =", format(metrics.accuracy_score(y_test, gnb_pred)))

#Saving model to disk
pickle.dump(lr,open('lr_model.pkl','wb'))
pickle.dump(knn,open('knn_model.pkl','wb'))
pickle.dump(gnb,open('gnb_model.pkl','wb'))

# Loding model to compare the result 
lr_model = pickle.load(open('lr_model.pkl','rb'))
knn_model = pickle.load(open('knn_model.pkl','rb'))
gnb_model = pickle.load(open('gnb_model.pkl','rb'))
