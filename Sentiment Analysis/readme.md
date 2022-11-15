# Snowpark Sentiment Analysis Example

## 1. Overview

This example is intended to demonstrate the use of Snowpark Python and Dynamic File Access to build a sentiment analysis model using product reviews. We'll then use this model to determine the sentiment if new data imported from an unstructured file in a Snowflake Stage.

## 2. What you'll learn  

- Get an understanding on how to implement an end-to-end ML pipeline using Snowpark for Python.
- Develop using Snowpark for Python API, and Snowpark for Python UDFs, vectorized UDFs and Stored Procedures.
- Data Exploration, visualization and preparation using Python popular libraries.
- Machine Learning using scikit-learn python package
- Deploying and using an ML model for scoring in Snowflake using Snowpark for Python.

## 4. Usage/Steps

* Step 1: Run through the Sentiment Analysis Notebook. This will upload the required files to Snowflake, and create the database needed for this demo. Make sure to customize creds.json
* Step 2: You can now run the Sentiment Analysis Demo.

If you are using icetire, throughout this process, if you see errors about Python packages not found in your conda environment, you can simply install them through a pip command directly in the Setup notebook as shown in the Section 2. Python Librairies.