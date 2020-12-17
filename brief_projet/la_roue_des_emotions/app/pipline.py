
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, ComplementNB, BernoulliNB
from  sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from time import time
from collections import defaultdict
from joblib import dump

from time import time
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, ComplementNB, BernoulliNB
from sklearn.decomposition import FastICA, KernelPCA, TruncatedSVD, SparsePCA, NMF, FactorAnalysis, LatentDirichletAllocation

import nltk
import pandas as pd
import numpy as np
import nltk
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
from sklearn import svm
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from  sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score

df =  pd.read_csv("./data/Emotion_final.csv")


stop_words = nltk.corpus.stopwords.words("english")

corpus = np.array(df['Text'])
targets = np.array(df['Emotion'])



pipe10 = Pipeline([
    ('vect', CountVectorizer(stop_words = stop_words, ngram_range = (1, 2))),
    ('sgd', SGDClassifier()),
])

pipe20 = Pipeline([
    ('vect', CountVectorizer(stop_words = stop_words, ngram_range = (1, 2))),
    ('mult_nb', MultinomialNB()),
])
pipe30 = Pipeline([
    ('vect', CountVectorizer(stop_words = stop_words, ngram_range = (1, 2))),
    ('bern_nb', BernoulliNB()),
])
pipe50 = Pipeline([
    ('vect', CountVectorizer(stop_words = stop_words, ngram_range = (1, 2))),
    ('sgd', SGDClassifier()),
])
pipe60 = Pipeline([
    ('vect', CountVectorizer(stop_words = stop_words, ngram_range = (1, 2))),
    ('tfidf', TfidfTransformer()),
    ('svml', LinearSVC()),
])
pipe70 = Pipeline([
    ('vect', CountVectorizer(stop_words = stop_words, ngram_range = (1, 2))),
    ('tfidf', TfidfTransformer()),
    ('bern_nb', BernoulliNB()),
])
def run_pipes(pipes, splits=10, test_size=0.2, seed=0):  
    res = defaultdict(list)
    spliter = ShuffleSplit(n_splits=splits, test_size=test_size, random_state=seed)
    for idx_train, idx_test in spliter.split(corpus):
        for pipe in pipes:
            # name of the model
            name = "-".join([x[0] for x in pipe.steps])
            
            # extract datasets
            X_train = corpus[idx_train]
            X_test = corpus[idx_test]
            y_train = targets[idx_train]
            y_test = targets[idx_test]
            
            # Learn
            start = time()
            pipe.fit(X_train, y_train)
            fit_time = time() - start
            
            # predict and save results
            y = pipe.predict(X_test)
            res[name].append([
                fit_time,
                f1_score(y_test, y, average = 'micro'),
                precision_score(y_test, y,average='micro'),
                recall_score(y_test, y, average='micro')
                
            ])
    return res
def print_table(res):
    # Compute mean and std
    final = {}
    for model in res:
        arr = np.array(res[model])
        final[model] = {
            "name" : model, 
            "time" : arr[:, 0].mean().round(2),
            "f1_score": arr[:,1].mean().round(3),
            "Precision" : arr[:,2].mean().round(3),
            "Recall" : arr[:,3].mean().round(3)
        }
    df = pd.DataFrame.from_dict(final, orient="index").round(3)
    return df





res = run_pipes([pipe10, pipe20, pipe30, pipe50 ,pipe60, pipe70], splits=1)
pickle.dump(res, open('filename.pkl', 'wb'))


df1 = pd.read_csv('data/text_emotion.csv')



stop_words = nltk.corpus.stopwords.words("english")

corpus1 = np.array(df1['content'])
targets1 = np.array(df1['sentiment'])

pipe0 = Pipeline([
    ('vect', CountVectorizer(stop_words = stop_words, ngram_range = (1, 2))),
    ('sgd', SGDClassifier()),
])

pipe2 = Pipeline([
    ('vect', CountVectorizer(stop_words = stop_words, ngram_range = (1, 2))),
    ('mult_nb', MultinomialNB()),
])
pipe3 = Pipeline([
    ('vect', CountVectorizer(stop_words = stop_words, ngram_range = (1, 2))),
    ('bern_nb', BernoulliNB()),
])
pipe5 = Pipeline([
    ('vect', CountVectorizer(stop_words = stop_words, ngram_range = (1, 2))),
    ('sgd', SGDClassifier()),
])
pipe6 = Pipeline([
    ('vect', CountVectorizer(stop_words = stop_words, ngram_range = (1, 2))),
    ('tfidf', TfidfTransformer()),
    ('svml', LinearSVC()),
])
pipe7 = Pipeline([
    ('vect', CountVectorizer(stop_words = stop_words, ngram_range = (1, 2))),
    ('tfidf', TfidfTransformer()),
    ('bern_nb', BernoulliNB()),
])
def run_pipes1(pipes, splits=10, test_size=0.2, seed=0):  
    res2 = defaultdict(list)
    spliter = ShuffleSplit(n_splits=splits, test_size=test_size, random_state=seed)
    for idx_train, idx_test in spliter.split(corpus1):
        for pipe in pipes:
            # name of the model
            name = "-".join([x[0] for x in pipe.steps])
            
            # extract datasets
            X_train1 = corpus1[idx_train]
            X_test1 = corpus1[idx_test]
            y_train1 = targets1[idx_train]
            y_test1 = targets1[idx_test]
            
            # Learn
            start = time()
            pipe.fit(X_train1, y_train1)
            fit_time = time() - start
            
            # predict and save results
            y = pipe.predict(X_test1)
            res2[name].append([
                fit_time,
                f1_score(y_test1, y, average = 'micro'),
                precision_score(y_test1, y,average='micro'),
                recall_score(y_test1, y, average='micro')
            ])
    return res2

def print_table1(res2):
    # Compute mean and std
    final = {}
    for model in res2:
        arr = np.array(res2[model])
        final[model] = {
            "name" : model,
            "time" : arr[:, 0].mean().round(2),
            "f1_score": arr[:,1].mean().round(3),
            "Precision" : arr[:,2].mean().round(3),
            "Recall" : arr[:,3].mean().round(3)
        }

    df1 = pd.DataFrame.from_dict(final, orient="index").round(3)
    return df1


# save models
res3 = run_pipes1([pipe0,pipe2, pipe3, pipe5 ,pipe6, pipe7], splits=1)
pickle.dump(res3, open('filename1.pkl', 'wb'))





