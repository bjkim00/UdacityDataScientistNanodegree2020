# Project 1 - Writing a Data Science Blog Post

## Intro

For this project, I created a machine learning pipeline to classify Distaster Relief Messages, which was eventually built into a Flask application.

## File Structure

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md

## Instructions to run

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


**Note: classifier.pkl and DisasterResponse.db files were not able to be uploaded to GitHub as the files were too big. However, if you run step 1 in the Instructions, you will end up creating these files in your local workspace, which then can be used for the run.py. Sorry for the inconvenience!**


