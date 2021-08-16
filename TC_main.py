# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 14:46:28 2021

@author: apsaros
"""

from TC_class import *
from nltk.corpus import reuters
from classifiers import *
#%%
def import_docs():
    """
    document loading function
    """
    documents = reuters.fileids()
     
    train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
    train_len = len(train_docs_id)
    docs_num = train_len
    train_docs_id = train_docs_id[:docs_num]
    test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))
    
    train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
    train_labels = [reuters.categories(doc_id) for doc_id in train_docs_id]
    
    test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]
    test_labels = [reuters.categories(doc_id) for doc_id in test_docs_id]
    
    return train_docs, train_labels, test_docs, test_labels

#%%
known_labels = False # whether we have labels or not (if not LDA is performed)
lda_thres = 0.1 if not known_labels else None # threshod distribution value for dominant topics thru LDA

# considered names of sklearn classifiers (see also separate BERT file)
classifiers = [
                'KN_classifier', 
                'MLP_classifier', 
                'LinearSVC_classifier', 
                'SVC_classifier', 
                'LogisticRegression_classifier', 
                'MultinomialNB_classifier'
               ]

# considered sklearn classifier classes (see also separate BERT file)
classifiers_cls = [
                KN_classifier, 
                MLP_classifier, 
                LinearSVC_classifier, 
                SVC_classifier, 
                LogisticRegression_classifier, 
                MultinomialNB_classifier
               ]

for i, clsfr in enumerate(classifiers_cls):
    # load the train and test data
    train_docs, train_labels, test_docs, test_labels = import_docs()
    TC_inst = TC()
    
    # fit classifier
    best_params = TC_inst.fit(train_docs, train_labels
                , classifier = clsfr
                , grid_search = False
                , feat_extract = 'tf-idf'
                , max_features = 100000
                , labels = 'known' if known_labels else 'unknown'
                , lda_thres = lda_thres
                )
    
    # make predictions on test data
    test_preds = TC_inst.predict(test_docs)
    
    # test labels transformation
    if not known_labels:
        test_labels = TC_inst.ltransformer.get_dominant_topics(TC_inst.lda_vectorizer_0.transform(test_docs))
    else:
        test_labels = TC_inst.ltransformer.transform(test_labels)
    
    # write results on file
    f = open(classifiers[i] + "_tf-idf_UnknownLabels_results.txt", "w")
    f.write("best hyperparameters: " + str(best_params) + " \n")
    
    precision, recall, f1 = TC_inst.get_accuracy(test_preds, test_labels, average_type = 'micro')
    f.write("Micro-average quality numbers \n")
    f.write("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f} \n".format(precision, recall, f1))
    
    precision, recall, f1 = TC_inst.get_accuracy(test_preds, test_labels, average_type = 'macro')
    f.write("Macro-average quality numbers \n")
    f.write("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f} \n".format(precision, recall, f1))
    f.close()