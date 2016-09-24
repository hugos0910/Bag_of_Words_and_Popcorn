#!/usr/bin/python3
import util
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# Import data
print('Importing data...')
train = pd.read_csv('labeledTrainData.tsv', header = 0, delimiter = '\t', quoting = 3)
test = pd.read_csv('testData.tsv', header = 0, delimiter = '\t', quoting = 3)
num_train_reviews = train["review"].size
num_test_reviews = test["review"].size

# Clean training data
print("Cleaning and parsing the training set movie reviews...")
clean_train_reviews = []
for i in range(0, num_train_reviews):
    if( (i+1)%1000 == 0 ):
        print("Training review %d of %d" %(i+1, num_train_reviews))
    clean_train_reviews.append(util.review_to_words( train["review"][i]))

print ("Creating the bag of words...")
vectorizer = CountVectorizer(analyzer = "word", 
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 5000) 

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

# Clean testing data
print("Cleaning and parsing the testing set movie reviews...")
clean_test_reviews = []
for i in range( 0, num_test_reviews ):
    if( (i+1)%1000 == 0 ):
        print("Testing review %d of %d" %(i+1, num_test_reviews))
    clean_test_reviews.append(util.review_to_words(test["review"][i]))
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Training classifier
print('Training classifier and predicting test data...')
classifier_name = 'RF'
clf = RandomForestClassifier(n_estimators = 500, n_jobs = -1)
prediction = clf.fit(train_data_features, train['sentiment']).predict(test_data_features)

# Export data to csv file
print('Writing data to file...')
test_id= test['id']
df_prediction = pd.DataFrame(prediction, index = test_id, columns = ["sentiment"])
df_prediction_csv = df_prediction.to_csv('prediction_BOW_%s.csv' %classifier_name, index_label = ["id"], quoting = 3)