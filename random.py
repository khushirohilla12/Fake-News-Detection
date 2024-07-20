#importing the important libraries

import numpy as np
import pandas as pd
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#loading the file for applying the alogrithm
file=pd.read_csv("filter.csv")

#Defining the dependent and independent variable

x = file["article_content"]
y = file["labels"]
#splitting the training and testing 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

#convert the text to vectors
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

#importing the random forest classifier
from sklearn.ensemble import RandomForestClassifier
#fitting and tranform the training and testing in model
RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
RandomForestClassifier(random_state=0)
pred_rfc = RFC.predict(xv_test)
per=RFC.score(xv_test, y_test)
#printing the score of the model
print("accuracy of the model=",per)

print(classification_report(y_test, pred_rfc))


