import numpy as np
import pandas as pd


'''reading the dataset of the facebook posts content is fake or real''' 
data=pd.read_csv("news.csv")
'''preprocessing of the dataset using pandas and numpy'''
print("shape of the dataset",data.shape)

'''checking the dataset duplicate value and drop them by this function'''
print("After dropping the duplicate value ",data.drop_duplicates(inplace=True))
print(data.shape)
'''now checking the sum of the null values in the dataset'''
print("null value sum of the dataset ",data.isnull().sum())

'''accessing the name of the coulmns'''

print("Column's name of the dataset",data.columns)
'''checking the datatype of the columns'''
print("datatype of the columns of dataset",data.info())

'''now we can also check the unique values'''
data.index.is_unique
data.drop(columns =['article_title', 'date','location','source'],inplace=True)
data.columns
print(data)

'''for random shuffling pf dataset'''
data = data.sample(frac = 1)
data.head()

'''changing the index of the dataset'''
data.reset_index(inplace = True)
data.drop(["index"], axis = 1, inplace = True)
data.head()
data.to_csv("filter.csv")

