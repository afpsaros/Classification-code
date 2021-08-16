# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 14:12:38 2021

@author: apsaros
"""

"""
Classes for text classification
"""

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score,precision_score,recall_score
import numpy as np
from classifiers import *
from LDA_class import LDA
import copy

class FE:
    """
    feature extractor class
    methods considered: bag of words, tf-idf, LDA
    """
    def __init__(self, max_features, method):
        self.method = method
        if method == 'bag_of_words':
            self.vectorizer = self.countvec_def(max_features, CountVectorizer)
        elif method == 'tf-idf':
            self.vectorizer = self.countvec_def(max_features, TfidfVectorizer)
        elif method == 'LDA':
            self.vectorizer_0 = self.countvec_def(100000, CountVectorizer)
            self.vectorizer = LDA(n_topics = max_features)
         
    def countvec_def(self, max_features, vectorizer):
        """
        basic vectorizer definition  
        """
        return vectorizer(
                              stop_words='english', tokenizer=self.tokenize_def(stopwords.words("english")),
                              max_df=0.95, min_df=0.02,
                              max_features=max_features,
                             )
    
    @staticmethod
    def tokenize_def(stop_words):
        """
        stemming function
        """   
        def tokenize(text):
            min_length = 3
            words = map(lambda word: word.lower(), word_tokenize(text))
            words = [word for word in words if word not in stop_words]
            tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))   
            p = re.compile('[a-zA-Z]+');
            filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length,tokens))
            
            return filtered_tokens
        return tokenize
    
    def fit(self, X): 
        return self.vectorizer.fit(X)
    
    def transform(self, X):
        return self.vectorizer.transform(X)

class TC:
    """
    text classification class 
    """
        
    def labels_transformer(self, Y):
        """
        transforms multi-labels into [1, 0, 0, 1, 0] vectors
        """
        self.ltransformer = MultiLabelBinarizer()
        self.ltransformer.fit(Y)
        
        return self.ltransformer
        
    def fit(self, X, Y, 
            max_features = 100000,
            classifier = LinearSVC_classifier, 
            grid_search = False,
            feat_extract = 'bag_of_words',
            labels = 'known', 
            lda_n_topics = 5, 
            lda_thres = 0.1
            ):
        
        """
        fits classifier
        """
        
        self.feat_extract = feat_extract
        # define feature extractor instance
        self.fe_inst = FE(max_features, feat_extract)
        X_copy = copy.deepcopy(X)
        if feat_extract == 'LDA':
            # for LDA feature extraction we need to first vectorize with bag of words
            inter = self.fe_inst.vectorizer_0.fit_transform(X)
            self.fe_inst.fit(inter)
            X = self.fe_inst.transform(inter)

        else:
            X = self.fe_inst.vectorizer.fit_transform(X)

        if labels == 'known':
            Y = self.labels_transformer(Y).transform(Y)
            
        if labels == 'unknown':
            # for unknown labels we first perform LDA and then use the dominant topics as labels

            self.lda_vectorizer_0 = self.fe_inst.countvec_def(max_features, CountVectorizer)
            self.ltransformer = LDA(n_topics = lda_n_topics)
            inter = self.lda_vectorizer_0.fit_transform(X_copy)
            self.ltransformer.fit(inter)
            Y = self.ltransformer.get_dominant_topics(inter, thres = lda_thres)
            
        self.clf_inst, best_params = classifier(X, Y, grid_search = grid_search)
        self.clf_inst.fit(X, Y)
        
        return best_params
        
    def predict(self, X):
        """
        label assignment predictor
        """
        
        if self.feat_extract == 'LDA':
            # for LDA feature extraction we need to first vectorize with bag of words
            X = self.fe_inst.transform(self.fe_inst.vectorizer_0.transform(X))
        else:
            X = self.fe_inst.transform(X)

        Y = self.clf_inst.predict(X)
        
        return Y
        
    def get_accuracy(self, Y_pred, Y_true, average_type = 'micro'):
        """
        standard performance evaluation metrics
        """
        
        precision = precision_score(Y_true, Y_pred, average=average_type)
        recall = recall_score(Y_true, Y_pred, average=average_type)
        f1 = f1_score(Y_true, Y_pred, average=average_type)
        
        return precision, recall, f1        
