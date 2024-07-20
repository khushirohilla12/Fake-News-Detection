#IMPORTING THE IMPORTANT LIBRARIES 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#LOADING THE CSV FILE OF THE DATASET
file=pd.read_csv("filter.csv")
file.head(100)

#DEFINING THE INDEPENDENT AND DEPENDENT VARIABLE 
x = file["article_content"]
y = file["labels"]

#SPLITTING THE TRAINING AND TESTING 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

#CONVERT THE text into vectors

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train,y_train)

pred_lr=LR.predict(xv_test)
pre=LR.score(xv_test, y_test)
print(pre)
print(classification_report(y_test, pred_lr))
