# Disaster Response Pipeline Project
This project goal is to create a website that categorize messages in order to help better respond to disasters. The project contains an ETL data pipline (process_data.py) to prepare and clean the data, a machine learning model (train_classifier.py) that classifies messages, and a web app that accepts user input and displays the results.  
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv sqlite:///etldb.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db /disaster_response_pipeline_project/model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
