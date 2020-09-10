import time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (plot_confusion_matrix,
                             plot_precision_recall_curve, plot_roc_curve,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def main():
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    st.title("Binary Classification App")
    st.sidebar.title("Binary Classification App")
    st.markdown("Are Your Mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are Your Mushrooms edible or poisonous? 🍄")
    st.sidebar.subheader("choose classifier")
    class_names = ['edible', 'poisonous']
    classifier = st.sidebar.selectbox(
        "classifier", ("Support Vector Machine (svm)", "Logistic Regression", "Random Forest"))
    st.sidebar.subheader("Model Hyparpramaters")
    

    @st.cache(allow_output_mutation=True)
    def load_data():
        data = pd.read_csv("mushrooms.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(allow_output_mutation=True)
    def split(df):
        y = df.values[:, 0]
        x = df.values[:, 1:]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def show_data():
        st.subheader("mushroom data set (classification)")
        st.write(df)

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test,
                                  display_labels=class_names)
            st.pyplot()
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    # Support Vector Machine option
    if classifier == "Support Vector Machine (svm)":
        c = st.sidebar.number_input(
            "C (Regularization Paramater)", 0.01, 10.0, step=0.01, key='c')
        kernal = st.sidebar.radio("Kernal", ('rbf', 'linear'))
        gamma = st.sidebar.radio(
            "Gamma (Kernal Coefficient)", ('scale', 'auto'))
        
        metrics = st.sidebar.multiselect(
            "What Metrics to Plot?", ("Precision-Recall Curve", "ROC Curve", "Confusion Matrix"))
        if st.sidebar.button("classify"):
            st.subheader("Support Vector Machine (svm) results")
            model = SVC(C=c, kernel=kernal, gamma=gamma)
            
            with st.spinner('Wait for it...'):
                model.fit(x_train, y_train)
                
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)

                st.write("Accuracy: ",accuracy)
                st.write("precision: ",precision_score(y_test,y_pred,labels=class_names))
                st.write("recall: ",recall_score(y_test,y_pred,labels=class_names))
                plot_metrics(metrics)    

# Logistic Regression option
    if classifier == "Logistic Regression":
        c = st.sidebar.number_input(
            "C (Regularization Paramater)", 0.01, 10.0, step=0.01, key='c')
        max_itr =st.sidebar.slider("maximum number of iteration",100,500)
        
        metrics = st.sidebar.multiselect(
            "What Metrics to Plot?", ("Precision-Recall Curve", "ROC Curve", "Confusion Matrix"))
        if st.sidebar.button("classify"):
            st.subheader("Logistic Regression results")
            model = LogisticRegression(C=c, max_iter=max_itr)
            
            with st.spinner('Wait for it...'):
                model.fit(x_train, y_train)
                
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)

                st.write("Accuracy: ",accuracy)
                st.write("precision: ",precision_score(y_test,y_pred,labels=class_names))
                st.write("recall: ",recall_score(y_test,y_pred,labels=class_names))
                plot_metrics(metrics)

        

# Random Forest option
    if classifier == "Random Forest":
        n_est = st.sidebar.number_input("The Number of Trees of The Forest",100,5000,step=10)
        max_depth = st.sidebar.number_input("The Maximum Depth of The Tree",1,20,step=1)
        bootstrap = st.sidebar.radio("Bootsttrap Samples When Bulding Trees",("True","False"))

        metrics = st.sidebar.multiselect(
            "What Metrics to Plot?", ("Precision-Recall Curve", "ROC Curve", "Confusion Matrix"))
        if st.sidebar.button("classify"):
            st.subheader("Random Forest results")
            model = RandomForestClassifier(n_estimators=n_est,max_depth=max_depth,bootstrap=bootstrap,n_jobs=-1)
            
            with st.spinner('Wait for it...'):
                model.fit(x_train, y_train)
                
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)

                st.write("Accuracy: ",accuracy)
                st.write("precision: ",precision_score(y_test,y_pred,labels=class_names))
                st.write("recall: ",recall_score(y_test,y_pred,labels=class_names))
                plot_metrics(metrics)

    if st.sidebar.checkbox("show raw data", False):
        show_data()


if __name__ == '__main__':
    main()
