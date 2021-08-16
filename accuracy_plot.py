# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 16:11:03 2021

@author: apsaros
"""

"""
accuracy (precision) plotter for standard classification
"""
import matplotlib.pyplot as plt

classifiers = [
                'KN', 
                'MLP', 
                'LinearSVC', 
                'SVC', 
                'LogisticRegression', 
                'MultinomialNB',
                'BERT'
               ]

vecs = ['BOW', 'LDA', 'tf-idf', 'tf-idf_GS']

res = [
       [0.8688, 0.7972, 0.8637, 0.7080],
       [0.8621, 0.8772, 0.8719, 0.8423],
       [0.7322, 0.8876, 0.9251, 0.9251],
       [0.9568, 0.9091, 0.9539, 0.9539],
       [0.8387, 0.8647, 0.9553, 0.9282],
       [0.5590, 0.9826, 0.9195, 0.8973],
       [None, None,0.9368,None],
       ]

for i, r in enumerate(res):
    plt.plot(r, '_', label = classifiers[i], markersize=20, markeredgewidth=2)
plt.xticks([0,1,2,3], vecs)
plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.savefig('acc_plot', dpi = 300, bbox_inches="tight")
plt.show()