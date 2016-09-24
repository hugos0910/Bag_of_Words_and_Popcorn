# Bag of Words Meets Bags of Popcorn

## Goal and Motivation 
The goal of this competition is to predict the user sentiment of a movie based on his/her review.  This is a good introduction to gain experience and basic understandings of natural language processing (NLP).  All of the python scripts beside main.py are starter scripts for practice obtained from Kaggle.

## Cleaning Data
The data cleaning for this competition can be borken down into the following steps:

1. Combine the training and testing set together
2. Parse the "review" column using BeautifulSoup
  * remove unundesired HTML elements
  * remove any character that is not a letter
  * convert everything to lowercase
  * split into individual words
  * apply Porter Stemmer
  * join the words back together to make one string for each review
3. Apply tf-idf to obtain feature vectors
4. Split the combined data back to training and testing set
5. Pickle the data for future use

## Choosing Classifiers
There are three classifiers chosen for this problem:
  * Logistic Regression (LR)
  * Stochastic Gradient Descent (SGD)
  * Multinomial Naive Bayes (MNB)

I decided to combine these three by using the majority voting ensemble, and give them weights as the following based on performance:
  * LR - 2
  * SGD - 3 
  * MNB - 1

## Result
The Kaggle results obtained ast listed as followed:
  * Ensemble - 0.95966
  * SGD - 0.95929
  * LR - 0.95920
  * MNB - 0.93831

As we can see, the ensemble performed only slightly better than the individual algorithms.  I attempted to train an SVM classifier by reducing features to 60,000, but that still took too long and I had to stop the script.  The score obtained from using the ensemble method will put me at rank 82 out of 579 participants, if the competition is still active.

