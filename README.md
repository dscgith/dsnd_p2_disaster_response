--- 
# Disaster Response: predicting categories of messages during disasters

This data science project on data from FigureEight was done as part of the Data Scientist Nanodegree from Udacity

#### Content
+ [Libraries used](#libraries-used)  
+ [Motivation](#motivation)  
+ [Data](#data)  
+ [What has been done?](#what-has-been-done)  
+ [How to use...](#how-to-use)  
   + [File structure](#how-to-navigate-the-file-structure)
   + [Scripts](#how-to-use-the-scripts)
+ [Possible future Improvements](#possible-future-improvements)  
+ [Acknowledgements](#acknowledgements) 

---

### Libraries used
+ sys
+ re
+ pickle
+ sqlalchemy
+ Pandas
+ Numpy
+ sklearn
+ NLTK
+ Flask

Python Version 3.6.3 was used for this project

---

### Motivation
This project should demonstrate how to work with natural language processing on text messages and prediction of categories.  
Based on the NLP pipeline a classifier predicts to which category a message belongs to take further action on that information.

---

### Data
The data was provided as part of the project, originating from Figure Eight Inc.

| filename | file description |
| :-- | :-- |
| messages.csv | List of messages from different sources sent during disaster situations |
| categories.csv | Set of categories the messages from messages.csv are assigned to |

The two datasets can be combined based on the given ID.

---

### What has been done?
* The data has been analyzed and cleaned 
   * the feature "child_alone" has been dropped, since no message were tagged with that category
   * some rows, where the feature "related" had values of 2 instead of binary 0 or 1 were dropped
   * some more categories are quite unbalanced, but have been used without further processing for now
* Using NLKT, CountVectorizer and TfidfTransformer the messages have been prepared for use with classifier algorithms
* With GridsearchCV different settings of the NLP pipeline and different classifiers have been evaluated
* A standard DecisionTreeClassifier was found to be best in terms of metrics and time to train (due to the unbalanced dataset metrics have to be looked at with care)

---

### How to use

#### How to navigate the file structure
- app  
|- template  
| |- master.html  # _main page of web app_  
| |- go.html  # _classification result page of web app_  
|- __run.py*__  # _Flask file that runs app_

- data  
|- disaster_categories.csv  # _data to process_  
|- disaster_messages.csv  # _data to process_  
|- __process_data.py*__  # _script to clean and save data to database_  
|- DisasterResponseExample.db   # _example for database to save clean data to_  

- models  
|- __train_classifier.py*__  # _script to train a classifier with database data_  
|- classifierExample.pkl  # _example for saved model_   

- notebooks  
|- ETL Pipeline Preparation.ipynb # preparations to build process_data.py  
|- ML Pipeline Preparation.ipynb # preparations to build train_classifier.py  

(* files to run, see following section)

#### How to use the scripts
1. ETL Pipeline: process the data and store it in a SQLite database
    * _process_data.py_ takes 3 arguments:
       + filepath of message.csv
       + filepath of categories.csv
       + filepath where the database is to be created
    
    * Example command:  
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
 
2. ML Pipeline: load the SQLite database, train a DecisionTreeClassifier on the data and store the classifier in a pickle file
    * _train_classifier.py_ takes 2 arguments:
       + filepath of the database
       + filepath where the model is to be stored            
    * Example command:  
      `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
 
3. Use data in webapp:
    * To run the webapp locally, go to the _app_ folder and run:  
      `python run.py`
    * Go to http://0.0.0.0:3001/ to view the webapp

If filenames or filepaths are altered, changes to `run.py` are necessary to run the webapp.  

---

### Possible future Improvements
* Further feature engineering could be done to enhance the data to use for classifying
   * create new features relating to the characteristics of the messages
   * deal with the fact, that the data is severely unbalanced regarding some of the categories (i.e. almost no messages for some categories like "offer", "tools" and "shops")
* Evaluate other Classifier algorithms
* Customize the webapp interface further with more data visualizations

---

### Acknowledgements
Starter Code was provided by Udacity as part of the Data Science Nanodegree project, the data was provided by Figure Eight Inc.

---
