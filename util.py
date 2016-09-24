#!/usr/bin/python3

import numpy as np
import pandas as pd
import re, nltk
import nltk.data
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from nltk.stem.porter import *

def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review).get_text() 
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]   
    return( " ".join(meaningful_words))

def review_to_wordlist(raw_review, remove_stopwords = False):
    review_text = BeautifulSoup(raw_review).get_text() 
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.lower().split()
    if remove_stopwords:
      stops = set(stopwords.words("english"))
      words = [w for w in words if not w in stops]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return(words)

def review_to_sentences(raw_review, tokenizer, remove_stopwords = False):
  raw_sentences = tokenizer.tokenize(raw_review.strip()) # Strip white spaces
  sentences = []
  for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        # Otherwise, call review_to_wordlist to get a list of words
        sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
  return sentences 


def makeFeatureVec(words, model, num_features):
  # Function to average all of the word vectors in a given
  # paragraph
  #
  # Pre-initialize an empty numpy array (for speed)
  featureVec = np.zeros((num_features,),dtype="float32")
  #
  nwords = 0.
  # 
  # Index2word is a list that contains the names of the words in 
  # the model's vocabulary. Convert it to a set, for speed 
  index2word_set = set(model.index2word)
  #
  # Loop over each word in the review and, if it is in the model's
  # vocaublary, add its feature vector to the total
  for word in words:
      if word in index2word_set: 
          nwords = nwords + 1.
          featureVec = np.add(featureVec,model[word])
  # 
  # Divide the result by the number of words to get the average
  featureVec = np.divide(featureVec,nwords)
  return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
  # Given a set of reviews (each one a list of words), calculate 
  # the average feature vector for each one and return a 2D numpy array 
  # 
  # Initialize a counter
  counter = 0.
  # 
  # Preallocate a 2D numpy array, for speed
  reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
  # 
  # Loop through the reviews
  for review in reviews:
     #
     # Print a status message every 1000th review
     if counter%1000. == 0.:
         print ("Review %d of %d" % (counter, len(reviews)))
     # 
     # Call the function (defined above) that makes average feature vectors
     reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
         num_features)
     #
     # Increment the counter
     counter = counter + 1.
  return reviewFeatureVecs