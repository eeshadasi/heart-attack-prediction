from flask import Flask,render_template
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("C:\\Users\\HI\\Desktop\\heart.csv")
X=df.drop("output",axis=1)
y=df["output"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(C=0.1)
model.fit(X_train,y_train)
pickle.dump(model,open("model.pkl","wb"))