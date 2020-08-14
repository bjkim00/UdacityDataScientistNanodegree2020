import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_cleaned_relate', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Top 10 most popular categories
    most_popular_categories = df.drop(columns = ['id', 'message', 'original', 'genre'])
    most_popular_categories = most_popular_categories.sum().sort_values(ascending = False).head(10)
    
    # Percentage of times shelter vs water was requested when food was requested
    food_shelter = df.drop(columns = ['id', 'message', 'original', 'genre', 'related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'cold', 'direct_report', 'weather_related', 
       'floods', 'storm', 'fire', 'earthquake', 'other_weather'])
    food_requested = food_shelter[food_shelter['food'] == 1]
    shelter_requested = food_shelter[(food_shelter['food'] == 1) & (food_shelter['shelter'] == 1)]
    water_requested = food_shelter[(food_shelter['food'] == 1) & (food_shelter['water'] == 1)]
    food_shelter_req = len(shelter_requested) / len(food_requested)
    food_water_req = len(water_requested) / len(food_requested)
    plot_shelter_water = [food_shelter_req, food_water_req]
    plot_x = ['Shelter requested', 'Water requested']
        
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x = most_popular_categories.index,
                    y = most_popular_categories.values
                )
            ],
            
            'layout': {
                'title': 'Most Popular Categories (Based on total messages)',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x = plot_x,
                    y = plot_shelter_water
                )
            ],
            
            'layout': {
                'title': 'Percentage of Shelter vs Water Requested when Food was requested',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Category Requested"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()