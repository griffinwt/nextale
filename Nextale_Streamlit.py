import pickle
import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#https://docs.streamlit.io/en/stable/api.html#display-data

st.set_page_config(
    page_icon=':books:',
    initial_sidebar_state='auto'
)

st.title('Find your next adventure...')

page = st.sidebar.selectbox(
    'Select-A-Page',
    ('Overview', 'Books', 'Movies', 'Video Games')
)

#recommender function

def make_recs_new(query, wout=''):  #need to set lookup and recommender global variables prior to calling   
    try:
        query=query.lower() #lowercase entry, lowercase titles (only during search, below)
        titles = list(lookup[lookup['product_title'].map(lambda x: x.lower()).str.contains(query)]['product_title'])
        q = titles[0] #this is the item to search for

        if wout == '':           
            top10_prods = []
            num_prod_revs = []
            avg_prod_stars = []
            for key in list(recommender[q].keys())[1:11]:
                top10_prods.append(key)
                num_prod_revs.append(round(lookup[lookup['product_title']==key]['tot_prod_reviews'].mean()))
                avg_prod_stars.append(round(lookup[lookup['product_title']==key]['avg_prod_stars'].mean(), 2))
            final_output_df = pd.DataFrame(data = {
                'Recommended Items':top10_prods,
                'Total Reviews for Product':num_prod_revs,
                'Avg Product Star Rating(1-5)':avg_prod_stars
            }, index=range(1,11))
            return final_output_df            
            
        else:
            
            wout = wout.lower() #lowercase
            filtered_query = [] #make empty list
            for key in list(recommender[q].keys()):
                if wout not in key.lower(): #check if avoided keyword is in results
                    filtered_query.append(key)
            top10_prods = []
            num_prod_revs = []
            avg_prod_stars = []
            for item in filtered_query[1:11]:
                top10_prods.append(item)
                num_prod_revs.append(round(lookup[lookup['product_title']==item]['tot_prod_reviews'].mean()))
                avg_prod_stars.append(round(lookup[lookup['product_title']==item]['avg_prod_stars'].mean(), 2))
            final_output_df = pd.DataFrame(data = {
                'Recommended Items':top10_prods,
                'Total Reviews for Product':num_prod_revs,
                'Avg Product Star Rating(1-5)':avg_prod_stars
                }, index=range(1,11))
            return final_output_df
        
    except:
        return f'Sorry, "{query}" does not appear to be in the product database'

    
if page == 'Overview':
    st.subheader('Overview')
    st.write('''

Recommendation systems are ubiquitous in today's attention-driven online environment, whether used by an online retailer to recommend similar products or a media application to prompt binge-watching. These systems use a combination of well known and proprietary techniques to engage customers with customized recommendations based on their previous preference. In this project, I created a relatively simple version of one of these recommenders using customer reviews between 1995-2015 provided by Amazon. I specifically looked at products in the categories of books, movies, and video games. To reflect the narrative element in each of these products, I titled my project "Nextale". It is a tool to assist the user in finding their "next tale" based on something they enjoyed previously.

*in order to expedite search results and save memory, this application works off of a truncated version of the full recommender; for each product, the 100 most similar products are stored in memory. This means that theoretically, if a user's "does not include" search term is too broad, less than 10 similar products may be available which would compromise the resulting printout. While acknowledging this potentiality, I would like to stress that this error has never been observed in any tests of the application.


    ''')


elif page =='Books':
    st.subheader('Books')
    st.write('''
The book recommender was trained on 1,489,354 reviews consisting of 795,389 unique customers and 46,575 unique products.

Try it out for yourself:
    ''')

    #lookup, recommender = books_look, books_rec #set variables
    lookup = pd.read_pickle('./pickles/books_look_small.pkl')
    
    with open('./pickles/new_books_rec.pkl', 'rb') as f:
        recommender = pickle.load(f)

    query = st.text_input('Please enter a word or phrase to search": ', max_chars=50)

    wout = st.text_input('If you would like to exclude a term from your search, please enter it here": ', max_chars=50)

    recommendation = make_recs_new(query, wout)

    if type(recommendation) == str:
        st.write(recommendation) #if result is a string, print it
    else:
        st.table(recommendation) #if result is a df, show it


elif page =='Movies':
    st.subheader('Movies')
    st.write('''
The movie recommender was trained on 4,405,432 reviews consisting of 1,867,543 unique customers and 72,385 unique products.

Try it out for yourself:
    ''')

    #lookup, recommender = movies_look, movies_rec #set variables

    lookup = pd.read_pickle('./pickles/movies_look_small.pkl')

    with open('./pickles/new_movies_rec.pkl', 'rb') as f:
        recommender = pickle.load(f)

    query = st.text_input('Please enter a word or phrase to search": ', max_chars=50)

    wout = st.text_input('If you would like to exclude a term from your search, please enter it here": ', max_chars=50)

    recommendation = make_recs_new(query, wout)

    if type(recommendation) == str:
        st.write(recommendation) #if result is a string, print it
    else:
        st.table(recommendation) #if result is a df, show it


elif page =='Video Games':
    st.subheader('Video Games')
    st.write('''
The video game recommender was trained on 1,648,136 reviews consisting of 979,917 unique customers and 15,938 unique products.

Try it out for yourself:
    ''')

    #lookup, recommender = vg_look, vg_rec #set variables

    lookup = pd.read_pickle('./pickles/vg_look_small.pkl')

    with open('./pickles/new_vg_rec.pkl', 'rb') as f:
        recommender = pickle.load(f)

    query = st.text_input('Please enter a word or phrase to search": ', max_chars=50)

    wout = st.text_input('If you would like to exclude a term from your search, please enter it here": ', max_chars=50)

    recommendation = make_recs_new(query, wout)

    if type(recommendation) == str:
        st.write(recommendation) #if result is a string, print it
    else:
        st.table(recommendation) #if result is a df, show it