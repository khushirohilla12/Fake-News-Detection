#IMPORTING THE IMPORTANT LIBRARIES OF THE ALGORITHMS
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#LOADING THE DATASET 
file=pd.read_csv("filter.csv")

#DEFINING THE INDEPENDENT AND DEPENDENT VARIABLE
x = file["article_content"]
y = file["labels"]
#SPLITTING THE THE TRAINING AND TESTING 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)


#CONVERT THE TEXT TO VECTORS
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

#DEFINING ENTROPY 
clf_entropy=DecisionTreeClassifier(criterion='entropy',random_state=0)
clf_entropy.fit(xv_train,y_train)
prediction=clf_entropy.predict(xv_train)
print(prediction)
'''print(confusion_matrix(y_test,prediction))'''

#DECISION CLASSIFIER 
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
percentage=DT.score(xv_test, y_test)
#PRINTING THE ACCURACY SCORE OF THE TRAINING SET
print(percentage)
print(classification_report(y_test, pred_dt))


#IMPORTING THE MATPLOTLIB FOR PLOT THE DECISION TREE

from matplotlib import pyplot as plt
text_representation = tree.export_text(clf_entropy)
print(text_representation)
fig=plt.figure(figsize=(100,100))
_=tree.plot_tree(clf_entropy ,filled=True)
fig.savefig("dt.png")
