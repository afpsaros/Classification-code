# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 10:38:45 2021

@author: apsaros
"""

"""
token frequency plotter
"""

from nltk.corpus import reuters, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from yellowbrick.text import FreqDistVisualizer
import matplotlib.pyplot as plt
import copy
    
stop_words = stopwords.words("english")
def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in stop_words]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))   
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length,tokens))
    
    return filtered_tokens    

def remove_words(text, words):
    for word in words:
        text = text.replace(word, '')
        
    return text
#%%
# List of document ids
documents = reuters.fileids()
 
train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
train_len = len(train_docs_id)
docs_num = train_len
train_docs_id = train_docs_id[:docs_num]
test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))

train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
train_docs_cp = copy.deepcopy(train_docs)

words = ['000', 'said', 'vs', 'lt', 'qtr', 'mln', 'cts', '4th', 'pct']
for i, doc in enumerate(train_docs):
    train_docs[i] = remove_words(doc, words)
#%%
n_features = 100000
vectorizer = CountVectorizer(
                            # max_df=0.95, min_df=0.02,
                            max_features=n_features,
                            stop_words='english')

train_docs = vectorizer.fit_transform(train_docs)
#%%
features = vectorizer.get_feature_names()
visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(train_docs)
visualizer.show(outpath="token_freq_before.png", dpi = 300)
#%%
n_features = 100000
vectorizer = CountVectorizer(
                            # max_df=0.95, min_df=0.02,
                            max_features=n_features)

train_docs = vectorizer.fit_transform(train_docs_cp)
#%%
features = vectorizer.get_feature_names()
visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(train_docs)
visualizer.show(outpath="token_freq_after.png", dpi = 300)





