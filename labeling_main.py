# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 17:42:47 2021

@author: apsaros
"""

"""
weak supervision via labeling functions using Snorkel
"""

from nltk.corpus import reuters
import pandas as pd
#%% load train and test data

documents = reuters.fileids()
 
train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
train_len = len(train_docs_id)
docs_num = train_len
train_docs_id = train_docs_id[:docs_num]

train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
train_docs_df = pd.DataFrame(data = train_docs, columns = ['text'])
#%% labels considered
ABSTAIN = -1
FOOD = 0
REC = 1
#%%
from snorkel.labeling import LabelingFunction
from snorkel.labeling import LFAnalysis
from snorkel.labeling import PandasLFApplier
#%% keyword functions definition
def keyword_lookup(x, keywords, label):
    if any(word in x.text.lower() for word in keywords):
        return label
    return ABSTAIN

def make_keyword_lf(keywords, label = FOOD):
    return LabelingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )

keyword_1 = make_keyword_lf(keywords=['grain', 'wheat', 'coffee', 'cocoa', 'corn', 'tea', 'coconut', 'sugar', 'rice', 'oat', 'orange', 'soy', 'potato'])
keyword_2 = make_keyword_lf(keywords=['oil', 'zinc', 'silver', 'gold', 'iron', 'steel', 'fuel', 'platinum', 'nickel', 'aluminum', 'copper', 'coal'])

lfs = [
    keyword_1,
    keyword_2
]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=train_docs_df )

print(L_train)

print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())
#%% create new labels
lbls = L_train
res = []

for el in lbls:
    if el[0] == -1 and el[1] == -1:
        res.append('a')
    elif (el[0] == 0 and el[1] == 1)  or (el[0] == 1 and el[1] == 0):
        res.append('b')
    else:
        res.append('c')
        
print(sum([1 if el == 'a' else 0 for el in res]))
print(sum([1 if el == 'b' else 0 for el in res]))
print(sum([1 if el == 'c' else 0 for el in res]))

new_lbls = []

for el in lbls:
    if any(el == 0):
        new_lbls.append('FOOD')
    elif any(el == 1):
        new_lbls.append('REC')
    else:
        new_lbls.append('None')

