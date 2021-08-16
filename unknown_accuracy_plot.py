# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 16:11:03 2021

@author: apsaros
"""

"""
accuracy (precision) plotter for unknown labels
"""
import matplotlib.pyplot as plt

classifiers = [
                'KN', 
                'MLP', 
                'LinearSVC', 
                'SVC', 
                'LogisticRegression', 
                'MultinomialNB',
               ]

vecs = ['LDA & tf-idf', 'WS & tf-idf']

res = [
       [0.3716, 0.7247],
       [0.5168, 0.7915],
       [0.5334, 0.7961],
       [0.5617, 0.7966],
       [0.5691, 0.7854],
       [0.5178, 0.7227],
       ]

for i, r in enumerate(res):
    plt.plot(r, '_', label = classifiers[i], markersize=20, markeredgewidth=2)
plt.xticks([0,1], vecs)
plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.savefig('unk_acc_plot', dpi = 300, bbox_inches="tight")
plt.show()