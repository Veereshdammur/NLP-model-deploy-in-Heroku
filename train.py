import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
#from sklearn.externals import joblib
import pickle
import requests 


# read the csv file as pandas dataframe                         
df= pd.read_csv("data.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Features and Labels
df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
df.drop(['v1'], axis=1, inplace=True)
X = df['v2']
y = df['label']

# Extract Feature With CountVectorizer
cv = CountVectorizer()

# Fit the Data
X = cv.fit_transform(X) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = MultinomialNB()
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

# dump the files 
filename = 'nlp_model.pkl'
pickle.dump(cv, open('tranform.pkl', 'wb'))
pickle.dump(clf, open(filename, 'wb'))
