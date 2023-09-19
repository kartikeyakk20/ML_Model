import streamlit as st
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import string  
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier  
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

st.title("Machine Learning Models")
st.write("---")
st.write("## Upload a csv file")
nvb_score=0
svm_score=0
dtc_score=0
xgb_score=0
def func():
    print("Changed")

df=st.file_uploader("Upload a csv file", type=".csv")
print(df)
if df:
    df=pd.read_csv(df,encoding='ISO-8859-1')
    df=df.drop(df.iloc[:,2:],axis=1)
    df.rename({'v1': 'target', 'v2': 'message'}, axis=1, inplace=True)
    stopword_list = stopwords.words('english')
    def clean(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        text = [word for word in text.split() if word not in stopword_list]
        text = ' '.join(text)
        return text
    df.message = df.message.apply(clean)
    st.dataframe(df.head())
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(df.target)
    tfidf = TfidfVectorizer()
    features=df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(features,target, test_size = 0.2, random_state = 0)
    X_train=tfidf.fit_transform(X_train).toarray()
    X_test=tfidf.transform(X_test).toarray()
    print(X_train)
    classifier = GaussianNB()
    classifier.fit(X_train,y_train)
    y_pred1 = classifier.predict(X_test)
    nvb_score=accuracy_score(y_test,y_pred1)
    clf = svm.SVC()
    clf.fit(X_train,y_train)
    clf.fit(X_train, y_train)
    y_pred2 = clf.predict(X_test)
    svm_score=accuracy_score(y_test,y_pred2)
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train,y_train)
    y_pred3 = clf.predict(X_test)
    dtc_score=accuracy_score(y_test,y_pred3)
    xgb = XGBClassifier()
    xgb.fit(X_train,y_train)
    y_pred4 = clf.predict(X_test)
    xgb_score=accuracy_score(y_test,y_pred4)
model=st.selectbox("Select the model",["Naive Bayes","SVM","Decision Trees", "XGBoost"],on_change=func)
st.set_option('deprecation.showPyplotGlobalUse', False)
if model=="Naive Bayes":
    st.write(nvb_score)
    # ConfusionMatrixDisplay.from_predictions(y_test,y_pred1)
    # st.pyplot()
elif model=="SVM":
    st.write(svm_score)
    # ConfusionMatrixDisplay.from_predictions(y_test,y_pred2)
    # st.pyplot()
elif model=="Decision Trees":
    st.write(dtc_score)
    # ConfusionMatrixDisplay.from_predictions(y_test,y_pred3)
    # st.pyplot()
elif model=="XGBoost":
    st.write(xgb_score)
    # ConfusionMatrixDisplay.from_predictions(y_test,y_pred4)
    # st.pyplot()
