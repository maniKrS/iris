import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset.
iris_df = pd.read_csv("https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score = svc_model.score(X_train, y_train)

logisticRegression = LogisticRegression()
logisticRegression.fit(X_train,y_train)

randomForestClassifier = RandomForestClassifier(n_estimators=100)
randomForestClassifier.fit(X_train,y_train)

st.cache_data()
def prediction(SepalLength, SepalWidth, PetalLength, PetalWidth,model):
  species = model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
  species = species[0]
  if species == 0:
    return "Iris-setosa"
  elif species == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"
  
st.sidebar.title('Iris Flower Species Prediction App')

sapel_length = st.sidebar.slider('sapelLength',float(iris_df['SepalLengthCm'].min()), float(iris_df['SepalLengthCm'].max()))
sapel_width = st.sidebar.slider('sapelWidth',float(iris_df['SepalWidthCm'].min()), float(iris_df['SepalWidthCm'].max()))
patel_length = st.sidebar.slider('patelLength',float(iris_df['PetalLengthCm'].min()), float(iris_df['PetalLengthCm'].max()))
patel_width = st.sidebar.slider('patelWidth',float(iris_df['PetalWidthCm'].min()), float(iris_df['PetalWidthCm'].max()))

dropDownPicker = st.sidebar.selectbox('classifier',options=['SVC','RandomForestClassifier','LogisticRegression'])

if st.sidebar.button('predict'):
  if dropDownPicker == 'SVC':
    result =prediction(sapel_length,sapel_width,patel_length,patel_width,svc_model)
    st.write(result)
    st.write(score)
  elif dropDownPicker == 'RandomForestClassifier':
    result =prediction(sapel_length,sapel_width,patel_length,patel_width,randomForestClassifier)
    st.write(result)
    st.write(randomForestClassifier.score(X_train,y_train))
  
  elif dropDownPicker == 'LogisticRegression':
    result =prediction(sapel_length,sapel_width,patel_length,patel_width,logisticRegression)
    st.write(result)
    st.write(logisticRegression.score(X_train,y_train))
   
  
  
