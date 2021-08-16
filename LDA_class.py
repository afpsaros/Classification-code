# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:05:58 2021

@author: apsaros
"""

from sklearn.decomposition import LatentDirichletAllocation

class LDA:
    """
    LDA class for topic modeling
    """
    def __init__(self, n_topics, max_iter = 5):
        
        self.lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
        
    def fit(self, X):
        """
        fits LDA
        """
        self.lda.fit(X)
        
    def transform(self, X):
        """
        topic distribution predictor
        """
        return self.lda.transform(X)

    def get_dominant_topics(self, X, thres = 0.5):
        """
        returns dominant topics: the ones with topic distribution value above threshold
        """
        return [[1 if el2 >= thres else 0 for el2 in el1] for el1 in self.lda.transform(X)]
    
    def get_model(self):
        """
        model getter
        """
        return self.lda
        