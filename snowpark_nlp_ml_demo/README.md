# Sentiment Analysis NLP with Snowpark ML

## Introduction
This demo is showing how we can do Natural Language Processing (NLP) and ML within 100% Snowflake using Snowpark Python.
Our use case is about doing : Sentiment Analysis with 100% Snowpark (feature engineering, train and prediction).

![image](https://user-images.githubusercontent.com/109098925/205313734-cf66fa17-587d-4a6b-8562-94150c604d36.png)

The origin of the dataset i used : https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data

## Snowflake Architecture :
<img width="600" alt="image" src="https://user-images.githubusercontent.com/109098925/205314615-996a1d86-5fb0-47bc-97d4-5729079837e1.png">

## Requirements
* Git
* Snowflake Account
* Snowpark for Python Environment

## Installation
* Clone this repository
* Unzip the **data/Dataset.zip**
* Run the SQL script **sql/init_database.sql** to create **IMDB** Database and the 2 tables : **TRAIN_DATASET**, **TEST_DATASET**
As an overview : 
![image](https://user-images.githubusercontent.com/109098925/199545072-f7151512-8f6f-4814-ab79-634f704ffc41.png)

* Then you need to load the 2 CSV files into Snowflake.
* If you don't have a snowpark env yet, use this yaml file : **snowpark-env/conda-env_snowpark_080.yml**
* Create and complete the Snowflake connexion in the file **config.py**
* Finally you can run the Notebook : **Sentiment_Analysis_NLP_with_Snowpark_ML.ipynb**

## Usage
You can easily run this demo locally with a Jupyter Notebook (Anaconda) for instance.

## Documentation
* **Deck presentation :** https://docs.google.com/presentation/d/1FxwnTC4xFkFOjZFxI3LHTwmk1ayVOGOrT4NxQ_e41rc/edit#slide=id.g13455ed4f81_0_2616
* **Video :** https://snowflake.zoom.us/rec/share/u2M24RKd6CkfO7w6JIUDEnB_w2xwd6UnvPL4MgJklBnifEcLBixPVmERyMHEOhSQ.idM77Qni74SRTAV_  	(Passcode: %e1ayYQ*)
* **Blogs :** https://medium.com/@ilyesmehaddi/natural-language-processing-nlp-and-ml-within-100-snowflake-using-snowpark-python-43e654111319
