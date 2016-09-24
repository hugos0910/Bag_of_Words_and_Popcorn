#!/usr/bin/python3

import numpy as np
import pandas as pd
import re, util
import nltk.data
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      

# Import data
print('Importing data...')
train = pd.read_csv('labeledTrainData.tsv', header = 0, delimiter = '\t', quoting = 3)
test = pd.read_csv('testData.tsv', header = 0, delimiter = '\t', quoting = 3)
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )

# Load trained model
model_name = "300features_40minwords_10context"
print('Loading trained model %s...' %model_name)
model = Word2Vec.load(model_name)

# Create feature vectors
print ("Creating average feature vecs for training reviews...")
clean_train_reviews = []
for review in train["review"]:
  clean_train_reviews.append(util.review_to_wordlist( review, remove_stopwords=True ))

trainDataVecs = util.getAvgFeatureVecs(clean_train_reviews, model, num_features)

print ("Creating average feature vecs for test reviews...")
clean_test_reviews = []
for review in test["review"]:
  clean_test_reviews.append(util.review_to_wordlist(review, remove_stopwords=True))

testDataVecs = util.getAvgFeatureVecs(clean_test_reviews, model, num_features)

# Training classifier
print('Training classifier and predicting test data...')
classifier_name = 'RF'
clf = RandomForestClassifier(n_estimators = 500, n_jobs = -1)
prediction = clf.fit(trainDataVecs, train['sentiment']).predict(testDataVecs)

# Export data to csv file
print('Writing data to file...')
test_id= test['id']
df_prediction = pd.DataFrame(prediction, index = test_id, columns = ["sentiment"])
df_prediction_csv = df_prediction.to_csv('prediction_Word2VecAvg_%s.csv' %classifier_name, index_label = ["id"], quoting = 3)