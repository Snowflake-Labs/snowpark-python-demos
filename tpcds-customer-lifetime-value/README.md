# Snowpark Python - TPC DS  - Customer Lifetime Value

This demo utilizes the [TPC DS sample](https://docs.snowflake.com/en/user-guide/sample-data-tpcds.html) dataset that is made available via  Snowflake share. It can be configured to run on either the 10 TB or the 100 TB version of the dataset.

This illustrates how to utilize Snowpark for feature engineering, training, and inference to answer a common question for retailers: What is the value of a customer across all sales channels?

&nbsp;  
## Setup

The TPC DS data is available already to you in your Snowflake account as shared database utlizing Snowflake's data sharing. This means you as the user will never incur the costs of storing this large dataset.

 1. Create a conda environment using the provided *environment.yml* file.
    1. `conda env create -f environment.yml `
    2. Activate that created conda environment by `conda activate pysnowpark_ml_tpcds`
 2. Edit the *creds.json* file to with your account information to connect to your account.
 3. Load Jupyter or equivalent notebook to begin executing the notebook.

&nbsp;  
## Snowflake ML

Snowpark ML Modeling is a collection of Python APIs for preprocessing data and training models. By performing these tasks within Snowflake, Snowpark ML lets you:

1. Transform your data and train your models without moving your data out of Snowflake.
2. Work with APIs similar to those you’re already familiar with, such as scikit-learn.
3. Keep your ML pipeline running within Snowflake’s security and governance frameworks.
4. Take advantage of the performance and scalability of Snowflake’s data warehouses.

The Snowpark ML Modeling package described here provides estimators and transformers that are compatible with those in the scikit-learn, xgboost, and lightgbm libraries. You can use these APIs to build and train machine learning models within Snowflake.

For a quick introduction to Snowpark ML Modeling, see our Quickstart -
https://quickstarts.snowflake.com/guide/intro_to_machine_learning_with_snowpark_ml_for_python/#0  

&nbsp; 