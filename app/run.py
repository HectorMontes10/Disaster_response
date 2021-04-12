import json
import plotly
import pandas as pd
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import joblib
import plotly.graph_objs as go

#Import custom class and functions:

import sys
sys.path.append("../models")
from starting_verb_extractor import StartingVerbExtractor
from calculate_textlength import CalculateLengthText
from dummy_estimator  import DummyEstimator
import tokenize_
import plotly_wc
from plotly_wc import plotly_wordcloud
from tokenize_ import tokenize

# load data

engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Cleaned_messages', engine)

# Create a wordcloud using plotly based on training messages (This may take a few minutes)

init_string = " ".join(tokenize(df['message'].iloc[0]))
for i in range(1,df.shape[0]):
    str_ = " ".join(tokenize(df['message'].iloc[i]))
    init_string = " ".join([init_string,str_])

fig = plotly_wordcloud(init_string)

#Custom title and annotations for wordcloud figure:

annotation = "Use the buttons in the upper right corner to interact with this WordCloud"
title = 'WordCloud for most tagged words'
fig.update_layout(
    title=go.layout.Title(
        text=title,
        x=0.5
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text=annotation,
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    )
)
fig.update_xaxes(showgrid=False,showticklabels=False, zeroline=False)

# load model

model = joblib.load("../models/classifier.pkl")
model = model.best_estimator_

#Star flask app 

app = Flask(__name__)

# index webpage displays cool visuals and receives user input text for model

@app.route('/')
@app.route('/index')

def index():
    
    # Extract data needed for visuals:
    
    #For first figure:
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #For second figure:
    
    cases_by_categories = df.iloc[:,4:].apply(sum)
    cases_by_categories = cases_by_categories.sort_values(axis=0, ascending=False)
    less_frequent_cases = cases_by_categories[-5:]
    less_frequent_names =  [x.capitalize().replace('_', ' ') for x in list(less_frequent_cases.index)]
    
    #For third figure:
    
    most_frequent_cases = cases_by_categories[0:5]
    most_frequent_names =  [x.capitalize().replace('_', ' ') for x in list(most_frequent_cases.index)]
        
    # create visualization:
    #--counts of messages by Genres
    #--Most and less frequent cases by categories
    #--wordcloud based on tokens extracted
    
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
                    x=most_frequent_names,
                    y=most_frequent_cases
                )
            ],

            'layout': {
                'title': 'Most frequent cases by categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=less_frequent_names,
                    y=less_frequent_cases
                )
            ],

            'layout': {
                'title': 'Less frequent cases by categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        fig
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