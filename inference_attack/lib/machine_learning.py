import csv
import glob
import math
import os
import pickle
import sys
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.naive_bayes import GaussianNB

# import tools
import lib.build_datasets as build_datasets
import lib.feature_extraction as build_datasets

def naive_bayes(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train) 
    y_pred = gnb.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)

    return [accuracy, y_pred]

def svm(X_train, X_test, y_train, y_test, kernel='linear'):
    clf = sklearn.svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)

    return [accuracy, y_pred]

def k_nearest_neighbors(X_train, X_test, y_train, y_test, k=10):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)

    return [accuracy, y_pred]

def random_forests(events_all_devices, X_train, X_test, y_train, y_test, n_estimators=100, random_state=0):
    clf=RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    feature_importance = pd.Series(clf.feature_importances_,index=feature_extraction.get_list_of_feature_names(events_all_devices)).sort_values(ascending=False)

    return [accuracy, y_pred, feature_importance]

def plot_confusion_matrix(y_true, y_pred, 
                          normalize=True,
                          title=True):

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(9, 8), dpi= 80)
    ax.imshow(cm, interpolation='none', aspect='auto', cmap=plt.cm.Blues)


    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return cm, fig, ax
