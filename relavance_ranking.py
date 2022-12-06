from functions import summarize,similarity, cv_X_cosine_similarity
import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
from nltk.tokenize import RegexpTokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import streamlit as st

def questions(qa_df,selected_df):
    prod_id = list(selected_df['asin'].unique())[0]
    qa_df_filtered = qa_df[(qa_df['asin']==prod_id) & (qa_df['questionType']=='open-ended')]
    qa_df_filtered = qa_df_filtered[['question']]
    qa_df_filtered['parsed_text'] = qa_df_filtered['question'].apply(lambda x: summarize(x))
    questions  = list(qa_df_filtered['parsed_text'].unique())
    return questions

def filtered_qa_function(qa_df,df_filtered,prod_id,open_ended_question=None):
    qa_df_filtered = qa_df[(qa_df['asin']==prod_id) & (qa_df['questionType']=='open-ended')]
    qa_df_filtered = qa_df_filtered[['question']]
    if open_ended_question is not None:
        qa_df_filtered = qa_df_filtered.append({'question': open_ended_question}, ignore_index=True)
    qa_df_filtered.rename(columns = {'question':'reviewText'}, inplace = True)

    return qa_df_filtered



def similarity_with_questions(qa_df,df_filtered,prod_id,questions,open_ended_question=None):
    similarity_df = df_filtered.copy()
    qa_df_filtered = filtered_qa_function(qa_df,df_filtered,prod_id,open_ended_question)
    similarity_df = pd.concat([similarity_df[['reviewText']], qa_df_filtered], ignore_index=True)
    similarity_df.reset_index(inplace=True)
    similarity_df['parsed_text'] = similarity_df['reviewText'].apply(lambda x: summarize(x))
    qa_df_filtered['parsed_text'] = qa_df_filtered['reviewText'].apply(lambda x: summarize(x))
    df_similarity_cv = cv_X_cosine_similarity(similarity_df, 'parsed_text')
    output_df = similarity(df_similarity_cv, similarity_df, questions, 'cv_cosine_sim', 'parsed_text')
    return output_df

def relevance_function(output_df,question):
    relavence_df = output_df[output_df['parsed_text_1']==question][['parsed_text_2','cv_cosine_sim']]
    relavence_df.columns=['review','similarity_score']
    relavence_df_2 = output_df[output_df['parsed_text_2']==question][['parsed_text_1','cv_cosine_sim']]
    relavence_df_2.columns=['review','similarity_score']
    relavence_df = pd.concat([relavence_df, relavence_df_2], ignore_index=True)
    relavence_df = relavence_df.sort_values(by=['similarity_score'], ascending = False)
    return relavence_df