import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

model = joblib.load('iris_model.pkl')
iris = sns.load_dataset('iris')

st.title("Iris Flower Prediction App")

st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)

species = ['Setosa', 'Versicolor', 'Virginica']
st.subheader("Prediction")
st.write(f"The predicted species is **{species[prediction[0]]}**.")

st.subheader("Iris Dataset Overview")
fig, ax = plt.subplots()
sns.scatterplot(data = iris, x = "sepal_length", y = "sepal_width", hue = "species", ax = ax)
st.pyplot(fig)
