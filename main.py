#!/usr/bin/python3

import util
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
import pickle

# Import data
print('Importing data...')
train = pd.read_csv('labeledTrainData.tsv', header = 0, delimiter = '\t', quoting = 3)
test = pd.read_csv('testData.tsv', header = 0, delimiter = '\t', quoting = 3)
y = np.ravel(train['sentiment'])

# Clean data
test['sentiment'] = np.NaN
combined_data = pd.concat([train,test], ignore_index = True)

print("Cleaning and parsing combined data movie reviews...")
clean_reviews = []
for review in combined_data['review']:
  clean_reviews.append(" ".join(util.review_to_wordlist(review)))

tfv = TfidfVectorizer(min_df = 3,  
                      max_features = None, 
                      strip_accents = 'unicode', 
                      analyzer = 'word', 
                      token_pattern = r'\w{1,}',
                      ngram_range = (1, 4), 
                      use_idf = 1,
                      smooth_idf = 1,
                      sublinear_tf = 1,
                      stop_words = 'english')

tfv.fit(clean_reviews)
X_combined = tfv.transform(clean_reviews)
X_train = X_combined[:len(train)]
X_test = X_combined[len(train):]

# # Export pickled data
# print('Exporting pickled data...')
# pickle.dump(X_train, open("X_train.p", "wb"))
# pickle.dump(y, open("y.p", "wb"))
# pickle.dump(X_test, open("X_test.p", "wb"))

# # Import pickled data
# print('Importing pickled data...')
# X_train = pickle.load( open( "X_train.p", "rb" ) )
# y = pickle.load( open( "y.p", "rb" ) )
# X_test = pickle.load( open( "X_test.p", "rb" ) )

# Training and predicting
classifier_name = 'ensemble'
print('Training and predicting...')
clf1 = LogisticRegression(penalty = 'l2', C = 30, dual = True, random_state = 0)
clf2 = SGDClassifier(alpha = 8e-05, 
                     class_weight=None, 
                     epsilon = 0.1, 
                     eta0 = 0.0,
                     fit_intercept=True, 
                     l1_ratio=0.15, 
                     learning_rate='optimal',
                     loss='modified_huber', 
                     n_iter = 5, 
                     n_jobs = -1, 
                     penalty='l2',
                     power_t=0.5, 
                     random_state=0, 
                     shuffle=True, 
                     verbose=0,
                     warm_start=False)
clf3 = MultinomialNB()
ensemble_clf = VotingClassifier(estimators = [('LR', clf1), ('SGD', clf2), ('MNB', clf3)], 
                                voting = 'soft', 
                                weights = [2, 3, 1])
prediction = ensemble_clf.fit(X_train,y).predict_proba(X_test)[:,1]

# Export data for submission
print('Writing data to file...')
test_id= test['id']
df_prediction = pd.DataFrame(prediction, index = test_id, columns = ["sentiment"])
df_prediction_csv = df_prediction.to_csv('prediction_%s.csv' %classifier_name, index_label = ["id"], quoting = 3)

