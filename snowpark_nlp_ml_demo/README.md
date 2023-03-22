# Sentiment Analysis NLP with Snowpark ML
![NLP Streamlit APP](https://user-images.githubusercontent.com/109098925/226951646-6992c798-8c36-4a2b-b7d9-5d39329f630c.png)

## Introduction

This demo is showing how we can do Natural Language Processing (NLP) and ML within 100% Snowflake using Snowpark Python ans Streamlit as well
Our use case is about doing : Sentiment Analysis with 100% Snowpark (feature engineering, train and prediction).

![image](https://user-images.githubusercontent.com/109098925/205313734-cf66fa17-587d-4a6b-8562-94150c604d36.png)

The origin of the dataset i used : https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data

Overview of the data : 

![Screenshot 2023-03-22 at 16 10 21](https://user-images.githubusercontent.com/109098925/226949522-3b2e041a-17d6-4c35-9dd9-e54dca7a7197.png)

## Snowflake Architecture :

<img width="600" alt="image" src="https://user-images.githubusercontent.com/109098925/205314615-996a1d86-5fb0-47bc-97d4-5729079837e1.png">

## Snowflake Features

* Snowpark DataFrame API For Python
* UDF Function
* Store Procedure
* Streamlit

## Learning Objectives :

* Do ML on Snowflake using Snowpark
* Load data into Snowflake
* Transform your data using Snowpark DataFrame API
* Train a scikit-learn model using Store Procedure inside Snowflake
* Deploy a model using UDF Function
* Inference with UDF Function
* Use Streamlit with Snowpark

## Requirements

* Git
* Snowflake Account
* Snowpark for Python (3.8) Environment

## Installation

1. Clone thes repository : 
```bash
git clone https://github.com/snowflakecorp/gsi-workload-demo.git
```
2. Setup Python environment with **Snowpark** and **Streamlit** :
```bash
cd snowpark_nlp_ml_demo/
conda update conda
conda update python
conda env create -f ./snowpark-env/conda-env_nlp_ml_sentiment_analysis.yml  --force
```
3. Update the Snowflake connexion file : **connection.json**
4. Activate Python environment using conda :
```bash
conda activate nlp_ml_sentiment_analysis
```
Use this to deactivate environment :
```bash
conda deactivate
```

## Usage
1. **Streamlit App** :
With the Streamlit APP you can run all the Demo by following the Menu options step by step from Loading Data to the Inference. The App make you able to load the data from the local system to Snowflake.
![NLP Streamlit APP](https://user-images.githubusercontent.com/109098925/226951696-275f1bd4-e3e9-4adf-b857-a41070bbd30b.png)

Launch the App using this :
```bash
streamlit ./streamlit/run Sentiment_Analysis_APP.py
```

2. **Notebook** :
Once you have setup everything and lead the data into Snowflake you can easily run the Demo using the Notebook (**Sentiment_Analysis_NLP_with_Snowpark_ML.ipynb**) as well:
```bash
jupyter notebook
```

## Documentation

* **Deck presentation :** https://docs.google.com/presentation/d/1FxwnTC4xFkFOjZFxI3LHTwmk1ayVOGOrT4NxQ_e41rc/edit#slide=id.g13455ed4f81_0_2616
* **Blogs :** https://medium.com/snowflake/natural-language-processing-nlp-and-ml-within-100-snowflake-using-snowpark-python-43e654111319
