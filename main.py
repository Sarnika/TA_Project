import streamlit as st
import gzip
from pandas import json_normalize
import pandas as pd
from relavance_ranking import questions,similarity_with_questions,relevance_function
from sentiment_analysis import sentiment_analysis

@st.cache
def first_part():

    reviews_df = pd.DataFrame()
    ReviewFile,MetaDataFile  = 'reviews_file.json.gz','meta_data.json.gz'
    reviews_df = pd.read_json(ReviewFile,lines=True,compression='gzip')
    metadata_df = pd.read_json(MetaDataFile,lines=True,compression='gzip')
    reviews_df = reviews_df[['asin','reviewText','summary','verified','vote']]
    metadata_df = metadata_df[['asin','title']]
    df = pd.merge(metadata_df,reviews_df,on='asin',how='right')
    df = df.drop_duplicates()
    count_df = df.title.value_counts().to_frame().reset_index()
    count_df.columns=['title','count']
    products = list(count_df[count_df['count']>=50]['title'])

    qa_df = pd.DataFrame()
    QAFile = 'qa_file.json.gz'
    qa = gzip.open(QAFile, 'r')
    c = 0
    for l in qa:
        data = eval(l)
        temp_df = json_normalize(data)
        c = c + 1
        if c == 1:
            qa_df = temp_df
        else:
            qa_df = pd.concat([qa_df, temp_df])
    return {'products':products,'df':df,'qa_df':qa_df}

products = first_part()['products']
df = first_part()['df']
qa_df = first_part()['qa_df']
selected_product = st.sidebar.selectbox('Select Product', products)
st.header(selected_product)
selected_df = df[df['title']==selected_product]
st.metric(label="# Reviews", value=len(selected_df))
import altair as alt


first_part()
dataframe= sentiment_analysis(selected_df)['dataframe']
dataframe = dataframe.assign(flag = 'Negative')
dataframe.loc[dataframe['vader_compound_score'] > 0.33, 'flag'] = 'Neutral'
dataframe.loc[dataframe['vader_compound_score'] > 0.66, 'flag'] = 'Positive'
count_flag_df = dataframe['flag'].value_counts(normalize=True) * 100
count_flag_df = count_flag_df.to_frame()
count_flag_df = count_flag_df.round(1)
count_flag_df.reset_index(inplace=True)
negative = float(count_flag_df[count_flag_df['index']=='Negative']['flag'])
neutral = float(count_flag_df[count_flag_df['index']=='Neutral']['flag'])
positive = float(count_flag_df[count_flag_df['index']=='Positive']['flag'])



@st.cache
def summary():
    positive_summary = sentiment_analysis(selected_df)['pos_summary']
    negative_summary = sentiment_analysis(selected_df)['neg_summary']
    return {'positive_summary':positive_summary,'negative_summary':negative_summary}
summary =  summary()
tab1, tab2, = st.tabs(["Sentiment Analysis & Summarization", "Relavance Ranking"])
with tab1:
    with st.expander("Sentiment Distribution"):
        col1,col2,col3=st.columns(3)
        col1.metric(label="% Positive", value=positive)
        col2.metric(label="% Neutral", value=neutral)
        col3.metric(label="% Negative", value=negative)
    with st.expander("Summarization"):
        st.subheader('Top 10 Positive Reviews summary')
        st.markdown(summary['positive_summary'])
        st.subheader('Top 10 Negative Reviews summary')
        st.markdown(summary['negative_summary'])


prod_id = list(selected_df['asin'].unique())[0]
questions = questions(qa_df, selected_df)



@st.cache
def output_df():
    output_df = similarity_with_questions(qa_df, selected_df, prod_id, questions)
    return  output_df

output_df = output_df()

@st.cache
def output_df_open(open):
    output_df = similarity_with_questions(qa_df, selected_df, prod_id, questions,open)
    return  output_df
with tab2:
    with st.expander("Relevance Ranking"):


        question = st.selectbox('Question', questions)
        final_df = relevance_function(output_df, question)
        st.dataframe(final_df.head(3))



    with st.expander('Ask a question!'):
        open = st.text_input("Enter Question")
        open = ' '+open
        questions = questions + [open]
        output_df_open = output_df_open(open)
        final_df_open = relevance_function(output_df_open, open)
        st.dataframe(final_df_open.head(3))




