# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 16:12:07 2021

@author: apsaros
"""

"""
Code for BERT classification
"""

if __name__ == '__main__':
    import time
    import pandas as pd
    from nltk.corpus import reuters
    from sklearn.preprocessing import MultiLabelBinarizer
    from simpletransformers.classification import MultiLabelClassificationModel
    from sklearn.metrics import f1_score,precision_score,recall_score
    #%% load train and test data
    
    documents = reuters.fileids()
     
    train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))
    
    train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
    test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]
    #%% transform multi-labels into [1, 0, 0, 1, 0] vectors
    
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform([reuters.categories(doc_id) for doc_id in train_docs_id])
    test_labels = mlb.transform([reuters.categories(doc_id) for doc_id in test_docs_id])
    #%% create train and test pandas dataframes
    train_data = [[el1, el2] for el1, el2 in zip(train_docs, train_labels)]
    train_df = pd.DataFrame(train_data, columns=["text", "labels"])
    
    test_data = [[el1, el2] for el1, el2 in zip(test_docs, test_labels)]
    test_df = pd.DataFrame(test_data, columns=["text", "labels"])
    #%%
    # create BERT model
    model = MultiLabelClassificationModel(
        "roberta",
        "roberta-base",
        num_labels=90,
        args={
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "num_train_epochs": 5,
        },
        use_cuda= False
    )
    #%% train the model and make predictions
    
    model.train_model(train_df)
    predictions, _ = model.predict(test_docs)
    #%% performance evaluation
     
    precision = precision_score(test_labels, predictions, average='micro')
    recall = recall_score(test_labels, predictions, average='micro')
    f1 = f1_score(test_labels, predictions, average='micro')
     
    print("Micro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
     
    precision = precision_score(test_labels, predictions, average='macro')
    recall = recall_score(test_labels, predictions, average='macro')
    f1 = f1_score(test_labels, predictions, average='macro')
     
    print("Macro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
