import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import re
import string
file=pd.read_csv("filter.csv")
file.head(100)
x = file["article_content"]
y = file["labels"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train,y_train)

pred_lr=LR.predict(xv_test)
LR.score(xv_test, y_test)
print(classification_report(y_test, pred_lr))



x = file["article_content"]
y = file["labels"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
DT.score(xv_test, y_test)
print(classification_report(y_test, pred_dt))


x = file["article_content"]
y = file["labels"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
RandomForestClassifier(random_state=0)
pred_rfc = RFC.predict(xv_test)
RFC.score(xv_test, y_test)

def wordopt(article_content):
    article_content = article_content.lower()
    article_content = re.sub('\[.*?\]', '', article_content)
    article_content = re.sub("\\W"," ",article_content) 
    article_content = re.sub('https?://\S+|www\.\S+', '', article_content)
    article_content = re.sub('<.*?>+', '', article_content)
    article_content = re.sub('[%s]' % re.escape(string.punctuation), '', article_content)
    article_content = re.sub('\n', '', article_content)
    article_content = re.sub('\w*\d\w*', '', article_content)    
    return article_content

file["article_content"] = file["article_content"].apply(wordopt)
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"article_content":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["article_content"] = new_def_test["article_content"].apply(wordopt) 
    new_x_test = new_def_test["article_content"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]),output_lable(pred_DT[0]),                                              
                                                                                                              output_lable(pred_RFC[0])))
news = str(input())
manual_testing(news)


