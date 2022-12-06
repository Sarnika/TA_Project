import pandas as pd
# Import relevant packages and downloads
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy import sparse
from scipy.sparse import csc_matrix
import nltk
from nltk.tokenize import RegexpTokenizer
import string
import seaborn as sns
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from nltk.tokenize import RegexpTokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer



def summarize(x):
    summ = ''
    parser_review_text = PlaintextParser.from_string(x, Tokenizer('english'))
    summarizer_lsa = LsaSummarizer()
    for sentence in summarizer_lsa(parser_review_text.document, 5): # Enter number of sentences
        summ = summ + ' ' + str(sentence)
    return summ


def cv_X_cosine_similarity(df,column): # Create count vectors
    cv_vec = CountVectorizer()
    cv_vectorized = cv_vec.fit_transform(df[column])
    cv_sim = cosine_similarity(cv_vectorized, dense_output=False)
    cv_pairs_coo = cv_sim.tocoo() # Convert to coordinate format
    review_1 = pd.DataFrame(cv_pairs_coo.row) # Create three sub-dataframes from coo rows
    review_2 = pd.DataFrame(cv_pairs_coo.col) # columns
    cosine_sim = pd.DataFrame(cv_pairs_coo.data) # data

    # Conatenate into a single dataframe
    cv_sim_df = pd.concat([review_1, review_2, cosine_sim], axis = 1)
    cv_sim_df.columns = ('review_1', 'review_2', 'cv_cosine_sim') # Rename columns
    cv_sim_df = cv_sim_df.sort_values(by=['cv_cosine_sim'], ascending = False) # Sort
    # Drop all self-matches; view again
    cv_sim_df = cv_sim_df.drop(cv_sim_df[cv_sim_df['review_1'] == cv_sim_df['review_2']].index)
    cv_sim_df = cv_sim_df.reset_index(drop = True) # Reset index
    cv_sim_df = cv_sim_df.sort_values(by=['cv_cosine_sim'], ascending = False) # Sort
    cv_sim_df = cv_sim_df.iloc[::2, :]
    cv_sim_df = cv_sim_df.reset_index(drop = True)
    cv_sim_df = cv_sim_df.sort_values(by=['cv_cosine_sim'], ascending = False)
    return cv_sim_df

def similarity(df,df1,questions,sim,name):
    merged_df = pd.merge(df,df1,how='left',left_on='review_1',right_on='index')
    x_df = merged_df[[name,'review_2',sim]]
    x_df.rename(columns={name: name+'_1'}, inplace=True)
    merged_df = pd.merge(x_df,df1,how='left',left_on='review_2',right_on='index')
    x_df = merged_df[[name+'_1',name,sim]]
    x_df.rename(columns={name:name+'_2'}, inplace=True)
    reviewText_df = x_df[(x_df[name+'_1'].isin(questions)) | (x_df[name+'_2'].isin(questions))]
    both_df = x_df[(x_df[name+'_1'].isin(questions)) & (x_df[name+'_2'].isin(questions))]
    final_df = pd.concat([reviewText_df,both_df]).drop_duplicates(keep=False)

    return final_df