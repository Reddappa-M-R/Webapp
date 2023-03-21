import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from sklearn.preprocessing import StandardScaler





class LassoRegression:
    def __init__(self, alpha, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        m, n = X.shape
        theta = np.zeros(n)
        self.intercept_ = np.mean(y)
        X_centered = X - np.mean(X, axis=0)
        for iteration in range(self.max_iter):
            old_theta = theta.copy()
            for j in range(n):
                X_j = X_centered[:, j]
                y_pred = X_centered.dot(theta) + self.intercept_
                r = y - y_pred + theta[j] * X_j
                z = X_j.dot(X_j)
                if z == 0:
                    theta[j] = 0
                else:
                    theta[j] = np.sign(np.sum(X_j*r)) * max(0, np.abs(np.sum(X_j*r)) - self.alpha) / z
            self.intercept_ = np.mean(y - X_centered.dot(theta))
            if np.sum(np.abs(theta - old_theta)) < self.tol:
                break
        self.coef_ = theta
    
    def predict(self, X):
        return X.dot(self.coef_) + self.intercept_





st.title("Boston_Housing_Price")
st.header("Predicting Price")

df = pd.read_csv("data.csv")

pickle_in = open("lasso1.pkl","rb")
lasso=pickle.load(pickle_in)

nav = st.sidebar.radio("Navigation", ["Home","Prediction","Contribute"])
if nav == "Home":
    st.image("Housing.jpg", width = 500)
    if st.checkbox("Show table"):
        st.table(df)

    val = st.slider("Filter df using age",0,100)
    df=df.loc[df["age"]>=val]
    
    graph = st.selectbox("What kind of graph ?", ["With_age","with_tax"])
    if graph == "With_age":
        fig, ax = plt.subplots()
        ax.scatter(df['age'],df['medv'])
        plt.title('Scatter')
        plt.ylim(0)
        plt.xlabel("age")
        plt.ylabel("medv")
        st.pyplot(fig)

    if graph == "with_tax":
        fig, ax = plt.subplots()
        ax.scatter(df['tax'],df['medv'])
        plt.title('Scatter')
        plt.ylim(0)
        plt.xlabel("tax")
        plt.ylabel("medv")
        st.pyplot(fig)





if nav == "Prediction":
    st.header("Know your House Price")
    crim = st.number_input("Enter crim",0.01,0.10,step=0.01)
    zn = st.number_input("Enter zn",1.0,19.00,step=1.00)
    indus = st.number_input("Enter indus",1.0,15.00,step=1.00)
    chas = st.number_input("Enter chas",1.0,5.00,step=1.00)
    nox = st.number_input("Enter nox",0.01,1.00,step=0.01)
    rm = st.number_input("Enter rm",1.0,10.00,step=1.00)
    age = st.number_input("Enter your age",5.0,100.00,step=5.00)
    dis = st.number_input("Enter dis",1.0,10.00,step=1.00)
    rad = st.number_input("Enter rad",1.0,50.00,step=5.00)
    tax = st.number_input("Enter tax",100.0,500.00,step=50.00)
    ptratio = st.number_input("Enter ptratio",1.0,30.00,step=1.00)
    black = st.number_input("Enter black",100.0,500.00,step=50.00)
    lstat = st.number_input("Enter lstat",1.0,20.00,step=1.00)

    val = [[crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,black,lstat]]
    val = np.array(val)
    X_test = pd.DataFrame(val)

    # Scaling the dataset
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)


    
    pred = lasso.predict(X_test)[0]
    if st.button("Predict"):
        st.success(f"Your predicted medv is in $ {round(pred)}k")


if nav == "Contribute":
    st.header("Contribute to our dataset")
    ex = st.number_input("Enter your age",0.0,100.0)
    sal = st.number_input("Enter your medv",0.00, 50.00,step=1000.0)
    if st.button("Submit"):
        st.success("Submitted")

