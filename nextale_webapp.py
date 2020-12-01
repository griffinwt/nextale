#referred to GA DSI Week 6 lesson on flask

import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, Response, render_template, jsonify

# initialize the flask app
app = Flask('myApp')

### route 1: hello world
# define the route
@app.route('/')
# create the controller
def home():
    return "Hello, world!"

### route 2: return a "web page"
@app.route('/hc_page')
def hc_page():
    # return some hard-coded HTML
    return '<html><body><h1>This is a hard coded page!</h1><p>Here is some hard-coded content. Isn\'t it pretty?</p></body></html>'

### route 3: return some data
@app.route('/hc_page.json')
def json():
    # create some data to return as json
    best_stuff = {
        "coast": "east",
        "movie" : "The Matrix",
        "hairstyle" : "bald"
    }
    return jsonify(best_stuff), 200

### route 4: show a form to the user
@app.route("/form")
def form():
    # use flask's render_template function to display the html page
    return render_template("form.html")

@app.route("/submit")
def make_predictions():
    # load the form data from the incoming request
    user_input = request.args

    # coerce data into a format that we can pass to our model
    # data = [
    #     int(user_input['OverallQual']),
    #     int(user_input['FullBath']),
    #     int(user_input['GarageArea']),
    #     int(user_input['LotArea'])
    # ]
    # return jsonify({'data': data})

    data = np.array([
        int(user_input['OverallQual']),
        int(user_input['FullBath']),
        int(user_input['GarageArea']),
        int(user_input['LotArea'])
    ]).reshape(1,-1)

    model = pickle.load(open("model/model.p", "rb"))
    prediction = model.predict(data)[0]

    return render_template("results.html", uri = round(prediction, 2))

# run the app
if __name__ == '__main__':
    app.run(debug = True)
