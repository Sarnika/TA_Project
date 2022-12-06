import streamlit as st
import gzip
import json
import pandas as pd
from pandas import json_normalize
# Import relevant packages
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import opinion_lexicon
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, matthews_corrcoef, average_precision_score, roc_auc_score
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
from nltk.tokenize import RegexpTokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

from functions import summarize,similarity, cv_X_cosine_similarity


def sentiment_analysis(selected_df):
    vader = SentimentIntensityAnalyzer()
    # Use the engine and the VADER 'compound' score to iterate through rows and produce a VADER score
    # Can use 'positive', 'negative', and 'neutral' instead of 'compound'
    # Works on strings so no need to tokenize
    selected_df['reviewText'] = selected_df['reviewText'].astype(str)
    selected_df['vader_compound_score'] = [vader.polarity_scores(x)['compound'] for x in selected_df['reviewText']]
    selected_df.sort_values(by=['vader_compound_score'], ascending=False, inplace=True)

    summarizer_lsa = LsaSummarizer()
    pos_review_text = selected_df.head(10).reviewText.str.cat(sep=' ')
    parser_review_text = PlaintextParser.from_string(pos_review_text, Tokenizer('english'))
    # Use text object created above; sentence-based tokenizer
    pos_summary = ''
    for sentence in summarizer_lsa(parser_review_text.document, 5):  # Enter number of sentences
        pos_summary = pos_summary + " " + str(sentence)

    neg_review_text = selected_df.tail(10).reviewText.str.cat(sep=' ')
    neg_parser_review_text = PlaintextParser.from_string(neg_review_text, Tokenizer('english'))
    # Use text object created above; sentence-based tokenizer
    neg_summary = ''
    for sentence in summarizer_lsa(neg_parser_review_text.document, 5):  # Enter number of sentences
        neg_summary = neg_summary + " " + str(sentence)

    return {'dataframe':selected_df,'pos_summary':pos_summary,'neg_summary':neg_summary}
