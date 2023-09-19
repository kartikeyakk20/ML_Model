import streamlit as st
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import string  
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier  
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV


st.title("Machine Learning Models")
st.write("---")
st.write("## Upload a csv file")
if 'df' not in st.session_state:
    st.session_state.df=st.file_uploader("Upload a csv file", type=".csv")
    df=pd.read_csv(st.session_state.df,encoding='ISO-8859-1')
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
    # X_train, X_test, y_train, y_test = train_test_split(features,target, test_size = 0.2, random_state = 0)
    features=tfidf.fit_transform(features).toarray()
    kfoldv=KFold(5)
    # X_test=tfidf.transform(X_test).toarray()

    #NaiveBayesModel
if 'nvb_score' not in st.session_state:
    classifier = GaussianNB()
    parameters1={'var_smoothing': [1, 0.1, 0.01]}
    classifier1=GridSearchCV(classifier,param_grid=parameters1,scoring='accuracy',cv=kfoldv)
    classifier1.fit(features,target)
    st.session_state.nvb_score=classifier1.best_score_
    # y_pred1 = classifier.predict(X_test)

    #SVM
if 'svm_score' not in st.session_state:
    clf = svm.SVC()
    parameters2={'C': [1,2,3,4,5], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf','linear','poly','sigmoid']}
    clf1=GridSearchCV(clf,param_grid=parameters2,scoring='accuracy',cv=kfoldv)
    clf1.fit(features,target)
    st.session_state.svm_score=clf1.best_score_
    # clf.fit(X_train, y_train)
    # y_pred2 = clf.predict(X_test)

    #Decision Trees
if 'dtc_score' not in st.session_state:

    dtc = DecisionTreeClassifier()
    parameters3={'max_depth': range(2,10,3), 'min_samples_leaf': [5, 10, 20, 50, 100], 'criterion': ["gini", "entropy"]}
    dtc1=GridSearchCV(dtc,param_grid=parameters3,scoring='accuracy',cv=kfoldv)
    dtc1.fit(features,target)
    st.session_state.dtc_score=dtc1.best_score_

    #XGBoost
if 'xgb_score' not in st.session_state:
    xgb = XGBClassifier()
    parameters4={'max_depth':range(2,10,2),'min_child_weight':range(1,6,2)}
    xgb1=GridSearchCV(xgb,param_grid=parameters4,scoring='accuracy',cv=kfoldv)
    xgb1.fit(features,target)
    st.session_state.xgb_score=xgb1.best_score_

    model=st.selectbox("Select the model",["Naive Bayes","SVM","Decision Trees", "XGBoost"])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if model=="Naive Bayes":
        st.write(st.session_state.nvb_score)
        # ConfusionMatrixDisplay.from_predictions(y_test,y_pred1)
        # st.pyplot()
    elif model=="SVM":
        st.write(st.session_state.svm_score)
        # ConfusionMatrixDisplay.from_predictions(y_test,y_pred2)
        # st.pyplot()
    elif model=="Decision Trees":
        st.write(st.session_state.dtc_score)
        # ConfusionMatrixDisplay.from_predictions(y_test,y_pred3)
        # st.pyplot()
    elif model=="XGBoost":
        st.write(st.session_state.xgb_score)
        # ConfusionMatrixDisplay.from_predictions(y_test,y_pred4)
        # st.pyplot()
