# Building a supervised machine learning model to recognize real (ham) vs. spam SMS text messages using the sklearn libary

# Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords # preprocessing
from string import punctuation # preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# Importing spam.csv as a DataFrame
messages = pd.read_csv('spam.csv')
messages.dropna(axis=1, inplace=True)
messages.columns = ['label', 'message']
print(messages.head())

# Adding a label_# column and message character length column
mapping = {'ham':0, 'spam':1}
messages['label_#'] = messages['label'].map(mapping)

messages['message_length'] = [len(x) for x in messages['message']]

# Exploratory data analysis
print(messages.head())
print(messages.describe())
print(messages.groupby('label').describe())
    # Note: spam.csv has 5572 observations with 4825 real and 747 spam messages

sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
plt.figure(figsize = (12,8))
messages[messages['label'] == 'spam']['message_length'].plot(bins=20, kind='hist', label='Spam message', alpha=0.25, color='red')
messages[messages['label'] == 'ham']['message_length'].plot(bins=40, kind='hist', label='Real message', alpha=0.25, color='blue')
plt.xlabel('Message character length')
plt.legend()
plt.show()
    # Note: The character length tends to be shorter and have a higher standard deviation for real messages

# Preprocessing the messages to remove punctuation, stop words, etc.
def message_preprocessor(message):
    '''
    Text as input
    1. Removes all punctuation
    2. Removes all stop words
    3. Returns a list of preprocessed, clean text
    '''
    total_stopwords = stopwords.words('english') + ['im', 'doin', 'dont', 'u', 'ur']
    
    # Remove all punctuation and store remaining characters in a list
    message_no_punctuation = [char for char in message if char not in punctuation]
    
    # Join the remaining characters (w/o punctuation) into a string
    message_no_punctuation = ''.join(message_no_punctuation)
    
    # Remove all stop words and store remaining words in a list
    message_no_punctuation_no_stopwords = [word for word in message_no_punctuation.split() if word.lower() not in total_stopwords]
    message_no_punctuation_no_stopwords = ' '.join(message_no_punctuation_no_stopwords) # join the words into a string using a ' ' (space) between words
    return message_no_punctuation_no_stopwords

messages['clean_message'] = messages['message'].apply(message_preprocessor)
print(messages.head())

# Converting the messages from text to vectors
X = messages['clean_message']
y = messages['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

cv = CountVectorizer(ngram_range=(1,2), stop_words='english')
X_train_dtm = cv.fit_transform(X_train) # fit and transform training data into a document-term (sparse) matrix
X_test_dtm = cv.transform(X_test) # transform testing data into a document-term (sparse) matrix using the fitted vocabulary

# tfidf = TfidfTransformer() # instantiating the term frequency-inverse document frequency transformer
# tfidf.fit_transform(X_train_dtm) # input is the count matrix from CountVectorizer

# Building and evaluating a Multinomial Naive Bayes model
mnb = MultinomialNB()
mnb.fit(X_train_dtm, y_train)
y_pred = mnb.predict(X_test_dtm)

mnb_confusion_matrix = confusion_matrix(y_test, y_pred)
# print(mnb_confusion_matrix)
mnb_classification_report = classification_report(y_test, y_pred)
# print(mnb_classification_report)

y_pred_proba = mnb.predict_proba(X_test_dtm)[:,1]
mnb_auc_score = roc_auc_score(y_test, y_pred_proba)
# print(mnb_auc_score)

    # note: use a pipeline to combine steps
pipeline = Pipeline([('bag_of_words', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('model', MultinomialNB())])
pipeline.fit(X_train, y_train)
y_pred_pipeline = pipeline.predict(X_test)
pipeline_confusion_matrix = confusion_matrix(y_test, y_pred_pipeline)
print(pipeline_confusion_matrix)
pipeline_classification_report = classification_report(y_test, y_pred_pipeline)
print(pipeline_classification_report)

# Building and evaluating a Logistic Regression model
logreg = LogisticRegression(solver='liblinear')
logreg.fit(X_train_dtm, y_train)
y_pred_logreg = logreg.predict(X_test_dtm)

logreg_confusion_matrix = confusion_matrix(y_test, y_pred_logreg)
print(logreg_confusion_matrix)
logreg_classification_report = classification_report(y_test, y_pred_logreg)
print(logreg_classification_report)

y_pred_logreg_proba = logreg.predict_proba(X_test_dtm)[:,1]
logreg_auc_score = roc_auc_score(y_test, y_pred_logreg_proba)
print(logreg_auc_score)