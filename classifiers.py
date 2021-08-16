# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:44:13 2021

@author: apsaros
"""

"""
Definition of considered classifiers
Options considered:
1. Whether to perform grid search (only one hyperparameter is varied)
2. k values for k-fold cross-validation
3. Range of values for varied hyperparameter
"""

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import copy

def dict_fix(d):
    static_keys = copy.deepcopy(list(d.keys()))
                            
    for key in static_keys:
        d[key[11:]] = d.pop(key)
        
    return d

def KN_classifier(X, Y, grid_search = False, cv_fold = 5, cv_range = np.arange(1, 8)):
    if grid_search:
        classifier = KNeighborsClassifier()
        param_grid = {'n_neighbors': cv_range}
        gscv = GridSearchCV(classifier, param_grid, cv=cv_fold)
        gscv.fit(X, Y)
        
        return KNeighborsClassifier(**gscv.best_params_), gscv.best_params_
    else:
        return KNeighborsClassifier(), None
    
def MLP_classifier(X, Y, grid_search = False, cv_fold = 5, cv_range = [(el,) for el in np.arange(20, 101, 20)], 
                  max_iter = 1000):
    if grid_search:
        classifier = MLPClassifier(max_iter=max_iter)
        param_grid = {'hidden_layer_sizes': cv_range}
        gscv = GridSearchCV(classifier, param_grid, cv=cv_fold)
        gscv.fit(X, Y)
        
        return MLPClassifier(max_iter=max_iter, **gscv.best_params_), gscv.best_params_
    else:
        return MLPClassifier(), None

def LinearSVC_classifier(X, Y, grid_search = False, cv_fold = 5, cv_range = ['hinge', 'squared_hinge']):
    if grid_search:
        classifier = OneVsRestClassifier(LinearSVC())
        param_grid = {'estimator__loss': cv_range}
        gscv = GridSearchCV(classifier, param_grid, cv=cv_fold)
        gscv.fit(X, Y)
        gscv.best_params_ = dict_fix(gscv.best_params_)
        
        return OneVsRestClassifier(LinearSVC(**gscv.best_params_)), gscv.best_params_  
    else:
        return OneVsRestClassifier(LinearSVC()), None

def SVC_classifier(X, Y, grid_search = False, cv_fold = 5, cv_range = np.arange(3, 6)):
    if grid_search:
        classifier = OneVsRestClassifier(SVC(kernel = 'poly'))
        param_grid = {'estimator__degree': cv_range}
        gscv = GridSearchCV(classifier, param_grid, cv=cv_fold)
        gscv.fit(X, Y)
        gscv.best_params_ = dict_fix(gscv.best_params_)
        
        return OneVsRestClassifier(SVC(kernel = 'poly', **gscv.best_params_)), gscv.best_params_  
    else:
        return OneVsRestClassifier(SVC()), None
    
def LogisticRegression_classifier(X, Y, grid_search = False, cv_fold = 5, cv_range = ['l1', 'l2']):
    if grid_search:
        classifier = OneVsRestClassifier(LogisticRegression(solver = 'liblinear'))
        param_grid = {'estimator__penalty': cv_range}
        gscv = GridSearchCV(classifier, param_grid, cv=cv_fold)
        gscv.fit(X, Y)
        gscv.best_params_ = dict_fix(gscv.best_params_)
        
        return OneVsRestClassifier(LogisticRegression(solver = 'liblinear', **gscv.best_params_)), gscv.best_params_  
    else:
        return OneVsRestClassifier(LogisticRegression()), None

def MultinomialNB_classifier(X, Y, grid_search = False, cv_fold = 5, cv_range = [0., 1.]):
    if grid_search:
        classifier = OneVsRestClassifier(MultinomialNB())
        param_grid = {'estimator__alpha': cv_range}
        gscv = GridSearchCV(classifier, param_grid, cv=cv_fold)
        gscv.fit(X, Y)
        gscv.best_params_ = dict_fix(gscv.best_params_)
        
        return OneVsRestClassifier(MultinomialNB(**gscv.best_params_)), gscv.best_params_ 
    else:
        return OneVsRestClassifier(MultinomialNB()), None