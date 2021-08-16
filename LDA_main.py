# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 09:41:00 2021

@author: apsaros
"""


from LDA_class import LDA
from nltk.corpus import reuters, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import copy
from yellowbrick.text import TSNEVisualizer
from yellowbrick.text import DispersionPlot
from yellowbrick.features import Rank2D

def plot_top_words(model, feature_names, n_top_words, title):
    """
    top words plotter 
    """
    fig, axes = plt.subplots(2, 5
                              , figsize=(30, 15)
                             , sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.savefig('lda_topics', dpi = 300, bbox_inches="tight")
    plt.show()
  
stop_words = stopwords.words("english")
def tokenize(text):
    """
    stemming function  
    """
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in stop_words]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))   
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length,tokens))
    
    return filtered_tokens    

def remove_words(text, words):
    """
    specific word remover
    """
    for word in words:
        text = text.replace(word, '')
        
    return text
#%% load train and test data
documents = reuters.fileids()
 
train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
train_len = len(train_docs_id)
docs_num = train_len
train_docs_id = train_docs_id[:docs_num]

train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]

# remove unwanted frequent words
words = ['000', 'said', 'vs', 'lt', 'qtr', 'mln', 'cts', '4th', 'pct', '10']
for i, doc in enumerate(train_docs):
    train_docs[i] = remove_words(doc, words)

train_docs_cp = copy.deepcopy(train_docs)
#%% LDA fit and top words in each topic
n_features = 100000
vectorizer = CountVectorizer(
                            # max_df=0.95, min_df=0.02,
                            max_features=n_features,
                            stop_words='english')

train_docs = vectorizer.fit_transform(train_docs)
feature_names = vectorizer.get_feature_names()   

n_topics =  5
n_top_words = 20
#%%
lda = LDA(n_topics = n_topics)
lda.fit(train_docs)

plot_top_words(lda.get_model(), feature_names, n_top_words, 'Topics in LDA model')
#%% tSNE plot of topics with topic labels

dom_topics = lda.get_dominant_topics(train_docs, thres = 0.7)
topic_names = ['topic_' + str(i) for i in range(1, n_topics + 1)]
topic_labels = ['None' if next((i for i, x in enumerate(dom_topics[i]) if x), n_topics) == n_topics
                else topic_names[next((i for i, x in enumerate(dom_topics[i]) if x), n_topics)]
                for i in range(len(dom_topics))]
#%%
tsne = TSNEVisualizer()
tsne.fit(train_docs, topic_labels)
tsne.show(outpath="tsne_topics.png", dpi = 300)
#%% dispersion plot for specific words defined in target_words

target_words = ['trade', 'oil', 'company', 'shares', 'tonnes']

text = [doc.split() for doc in [reuters.raw(doc_id) for doc_id in train_docs_id]]

visualizer = DispersionPlot(
    target_words,
    colormap="Accent",
    title="Lexical Dispersion Plot, Broken Down by Topic Labels"
)
visualizer.fit(text, topic_labels)
visualizer.show(outpath="dispersion.png", dpi = 300)
#%% re-do vectorization in order to have only 20 features to be ranked
n_features = 20
vectorizer = CountVectorizer(
                            # max_df=0.95, min_df=0.02,
                            max_features=n_features,
                            stop_words='english')

train_docs = vectorizer.fit_transform(train_docs_cp)
feature_names = vectorizer.get_feature_names()
#%% feature ranking vizualized with topic labels

col_names = feature_names
df = pd.DataFrame.sparse.from_spmatrix(train_docs, columns = col_names)

visualizer = Rank2D(algorithm='pearson')

visualizer.fit(df, topic_labels)        
visualizer.transform(df)      
visualizer.show(outpath="rank2d.png", dpi = 300)    










