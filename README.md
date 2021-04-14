# Disaster Response Pipeline Project

### Why this repository?

For the effective response to disasters (earthquakes, floods, fires, etc.) it is important to coordinate the work between government entities and civil society.

This project uses a database of twitter messages that people, communities or institutions sent during disaster events. The messages were classified into 36 different categories. Below some categories in the list:

- request
- offer
- aid related
- medical help

...

- child alone
- refugees
- death
- storm
- fire
- earthquake

Using nlp techniques on the database of tagged messages, we train a model that allows classifying new messages, hoping that it will be useful to better coordinate efforts. Each message can be assigned to several categories since they are not exclusive, so this is a typical multi-label classification problem. To train the model we use gridsearch on a parameter space defined for two estimators: random forests and logistic regression. 

If a message is written in the input box of the web application, the best model chosen is used to classify it in the categories to which it belongs, and then this ouput can be used to route the message to the institution that should attend the event (this last one is not implemented but it is a potential use).

<img src="Gif_clasificaci%C3%B3n.gif" alt="Gif_clasificación.gif">

To understand the characteristics of the training dataset, three visualizations are offered: the 5 categories with the most messages, the 5 categories with the fewest messages, and a cloud of frequent words in the messages.

<img src="visualizaciones.gif" alt="Gif_visualizacion.gif">

I hope this project will serve as a starting point for a more specialized disaster response solution. You can also use it as an outline to solve a similar problem where you need to train a multiclass classification model for word processing.

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or localhost:3001

Make sure to create the rules in the firewall to allow http / TCP traffic and grant python permissions so that the execution of the app locally works correctly. Enable listening on port 3001.

Enjoy it

### Files:

The structure for this project is:

- app
   - template
      - master.html  # main page of web app
      - go.html  # classification result page of web app
   - run.py  # Flask file that runs app
- data
   - disaster_categories.csv  # data to process 
   - disaster_messages.csv  # data to process
   - process_data.py #python script usefull for preprocessing data (clean messages)
   - DisasterResponse.db   # Database where clean messages are stored
- models
   - train_classifier.py 
   - classifier.pkl  # saved model
   - calculate_textlength.py
   - dummy_estimator.py
   - plotly_wc.py
   - starting_verb_extractor.py
   - tokenize_.py"

Notes:

1. Deleting classifier.pkl model is possible but once this is done you will have to run train_classifier.py again to obtain a new trained model to be used by your application. Model training may take time, depending on the capabilities of your server. training_classifier.py implements gridsearch for tuning parameters, you can edit the parameter search space if you wish for more intensive or less intensive training.
2. DisasterResponse.db is a database that can be deleted, but once this is done it will be necessary to run process_data.py again to create a new clean messages database. You can edit the script to customize the cleanup tasks on the disaster_messages.csv and disaster_categories.csv files.
3. calculate_textlength.py and starting_verb_extractor.py are custom transformers that calculate the length of each message in the vector of training texts, and mark the texts that start with verb or RT (Retweet). These classes are used to add two more features to the training matrix. Feel free to add new transformers if you think those present are not quite suitable for prediction.
4. dummy_estimator.py implements a dummy class to allow injection of different estimators to the gridsearch method. Currently the training function implements random forest and logistic regression estimators with their respective search parameters. You can inspect the space parameter in more detail in the build_model function inside train_classifier.py
5. tokenize_.py is a custom function that properly tokenizes the words in each message. It uses the word_tokenize and WordNetLemmatizer functions from the nltk package and returns clean tokens for training.
6. For this function the credits are from PrashantSaikia, who implements a method to draw a word cloud using the plotly package. You can access the original resource in this [github repository](https://github.com/PrashantSaikia/Wordcloud-in-Plotly/blob/master/plotly_wordcloud.py)

### Requeriments:

- sklearn==0.24.1
- pandas==1.1.5
- ntlk==3.6.1
- sqlalchemy==1.4.7
- plotly==4.14.3
- flask==1.1.2
- wordcloud==1.8.1

