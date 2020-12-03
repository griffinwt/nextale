#referred to GA DSI Week 6 lesson on flask

import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, Response, render_template, jsonify
import requests

import re
import time
import random
import sys
pd.set_option('display.max_colwidth', None)

from sklearn.metrics.pairwise import pairwise_distances, cosine_distances, cosine_similarity
from scipy import sparse
from matplotlib import pyplot as plt
import utils as ut

# initialize the flask app
app = Flask('myApp') #creating an instance of the flask class

### route 1: hello world
# define the route
@app.route('/')  # "decorator" + "route" + "function"
# create the controller
def home():
    return "Hello, world!"

### route 2: return a "web page"
@app.route('/hc_page')
def hc_page():
    # return some hard-coded HTML
    return '<html><body><h1>This is a hard coded page!</h1><p>Here is some hard-coded content. Isn\'t it pretty?</p></body></html>'

### route 4: show a form to the user
@app.route("/form")
def form():
    # use flask's render_template function to display the html page
    return render_template("reco_form.html")

@app.route("/submit")
def make_predictions():
    # load the form data from the incoming request
    user_input = request.args
    #print(f'{user_input}', file=sys.stderr)

    def rec_search(category, query, wout='no'):

        if category.lower() == 'video games':
            lookup, recommender = pd.read_pickle('./pickles/videog_look.pkl'),pd.read_pickle('./pickles/videog_rec.pkl')
        elif category.lower() == 'movies':
            lookup, recommender = pd.read_pickle('./pickles/movies_look.pkl'), pd.read_pickle('./pickles/movie_rec.pkl') 
        elif category.lower() == 'books':
            lookup, recommender = pd.read_pickle('./pickles/books_look.pkl'), pd.read_pickle('./pickles/books_rec.pkl')
        else:
            return "Sorry, that wasn't one of the available categories"

        try:
            query=query.title() #uppercase first letter of each word in the query in case it's not entered that way
            titles = list(lookup[lookup['product_title'].str.contains(query)]['product_title'])
            q = titles[0] #this is the item to search for

            if wout.lower() == 'no':
                query_dict = dict(recommender.loc[q].sort_values())
                
                final_printout = ''
                for key in list(query_dict.keys())[1:11]:
                    final_printout += '\n' #add new line
                    final_printout += f'{key}' #print item name
                    final_printout += '\n'
                    final_printout += f"""
                    This item has {round(lookup[lookup['product_title']==key]['tot_prod_reviews'].mean())} reviews 
                    and a {round(lookup[lookup['product_title']==key]['avg_prod_stars'].mean(), 2)} average star rating"""


                return f'''

                Recommending items similar to: {q}
                This item has {round(lookup[lookup['product_title']==q]['tot_prod_reviews'].mean())} reviews
                and a {round(lookup[lookup['product_title']==q]['avg_prod_stars'].mean(), 2)} average star rating.

                Here are the 10 recommended items for you based on your search parameters:
                
                {final_printout}
                
                '''
                
            else:
                
                wout = wout.title() #capitlize first letters
                query_dict = dict(recommender.loc[q].sort_values())
                filtered_query = [] #make empty list
                for key, value in query_dict.items(): #index into dictionary of results
                    if wout not in key: #check if avoided keyword is in results
                        filtered_query.append((key, value)) #make list of results that DON'T include "wout" keyword
                
                final_printout = ''
                for item in filtered_query[1:11]:
                    final_printout += '\n' #add new line
                    final_printout += f'{item[0]}' #print item name
                    final_printout += '\n'
                    final_printout += f"""
                    This item has {round(lookup[lookup['product_title']==item[0]]['tot_prod_reviews'].mean())} reviews 
                    and a {round(lookup[lookup['product_title']==item[0]]['avg_prod_stars'].mean(), 2)} average star rating"""
                
                return f'''

                Recommending items similar to: {q}
                This item has {round(lookup[lookup['product_title']==q]['tot_prod_reviews'].mean())} reviews
                and a {round(lookup[lookup['product_title']==q]['avg_prod_stars'].mean(), 2)} average star rating.

                Here are the 10 recommended items for you based on your search EXCLUDING "{wout}":
                
                {final_printout}
                
                '''
        
        except:
            return f'Sorry, "{query}" does not appear to be in the product database'
    
    
    # coerce data into a format that we can pass to our model
    # data = [
    #     int(user_input['OverallQual']),
    #     int(user_input['FullBath']),
    #     int(user_input['GarageArea']),
    #     int(user_input['LotArea'])
    # ]
    # return jsonify({'data': data})

    #data = np.array([
        #int(user_input['OverallQual']),
        #int(user_input['FullBath']),
        #int(user_input['GarageArea']),
        #int(user_input['LotArea'])
    #]).reshape(1,-1)
    
    #rec_search(str(user_input))

    #model = pickle.load(open("model/model.p", "rb"))
    #prediction = model.predict(data)[0]

    user_input_list = [
        user_input["cat"],
        user_input["query"],
        user_input["wout"]
    ]
    #print(f'{user_input_list}', file=sys.stderr)
    #print(f'{user_input_list}')
#['category', 'query', 'wout']
    recommendations = rec_search(user_input_list[0], user_input_list[1], user_input_list[2])
    return render_template("reco_results.html", recs = recommendations)
    #return render_template("results.html", uri = print(user_input_list))
# run the app
if __name__ == '__main__':
    app.run(debug = True)
