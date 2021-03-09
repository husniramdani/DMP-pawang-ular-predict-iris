import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

st.write("""
# Pawang Ular
""")

st.sidebar.header('Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.0)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.0)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 3.0)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 1.0)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('input parameters')
st.write(df)

dataset = pd.read_csv("./data/iris.csv")

array = dataset.values
x = array[:,0:4]
y = array[:,4]

models = []
pred = []

models.append(('Decision Tree', DecisionTreeClassifier(max_depth = 3, random_state = 1)))
models.append(('Gaussian Naive Bayes', GaussianNB()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors = 3)))
models.append(('Logistic Regression', LogisticRegression(solver = 'newton-cg')))
models.append(('Linear SVC', SVC(kernel='linear')))

@st.cache
def predictionModel():
    for name, model in models:
        pred.append(model.fit(x,y).predict(df))

predictionModel()
idx = [models[i][0] for i in range(len(models))]
pred = pd.DataFrame(pred, columns = ['Prediction'], index = idx)

st.subheader('Prediksi untuk tiap klasifikasi')
st.write(pred)

st.subheader('Hasil Prediksi Paling banyak')
st.text(pred['Prediction'].value_counts().index[0])

st.text("""
# ░░░░░░░░░░░░░░░░░░░░░░░
# ░░░░░▄▀▀▀▄░░░░░░░░░░░░░
# ▄███▀░◐░░░▌░░░░░░░░░░░░
# ░░░░▌░░░░░▐░░░░░░░░░░░░
# ░░░░▐░░░░░▐░░░░░░░░░░░░
# ░░░░▌░░░░░▐▄▄░░░░░░░░░░
# ░░░░▌░░░░▄▀▒▒▀▀▀▀▄░░░░░
# ░░░▐░░░░▐▒▒▒▒▒▒▒▒▀▀▄░░░
# ░░░▐░░░░▐▄▒▒▒▒▒▒▒▒▒▒▀▄░
# ░░░░▀▄░░░░▀▄▒▒▒▒▒▒▒▒▒▒▀▄
# ░░░░░░▀▄▄▄▄▄█▄▄▄▄▄▄▄▄▄▄▄▀▄
# ░░░░░░░░░░░▌▌░▌▌░░░░░░░
# ░░░░░░░░░░░▌▌░▌▌░░░░░░░
# ░░░░░░░░░▄▄▌▌▄▌▌░░░░░░░
# ░░░░░░PAWANG ULAR░░░░░░░
# ███████████████████████
""")
