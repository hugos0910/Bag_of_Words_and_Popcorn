import pandas as pd
import re, logging
import nltk.data
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import word2vec

# Import data
print('Importing data...')
train = pd.read_csv('labeledTrainData.tsv', header = 0, delimiter = '\t', quoting = 3)
test = pd.read_csv('testData.tsv', header = 0, delimiter = '\t', quoting = 3)
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = []  # Initialize an empty list of sentences

print ("Parsing sentences from training set")
for review in train["review"]:
  sentences += util.review_to_sentences(review, tokenizer)

print ("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
  sentences += util.review_to_sentences(review, tokenizer)

print('The length of the sentences list is %d' %len(sentences))
# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print ("Training model...")
model = word2vec.Word2Vec(sentences, 
                          workers = num_workers, 
                          size=num_features, 
                          min_count = min_word_count, 
                          window = context, 
                          sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# Save model
model_name = "300features_40minwords_10context"
model.save(model_name)