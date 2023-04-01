# 2023 NCAA March Madness Predictions

In this series of notebooks, we'll demonstrate how to set up a Snowflake environment, ingest data, conduct feature engineering, prepare and train a machine learning model, and ultimately, predict the outcomes of the 2023 NCAA March Madness tournament using [Snowpark for Python](https://www.snowflake.com/en/data-cloud/snowpark/).

These examples showcase some of the work conducted for the [March Machine Learning Mania 2023](https://www.kaggle.com/competitions/march-machine-learning-mania-2023/data) Kaggle competition and are inspired by the article [Predicting the Unpredictable — March Madness using Snowpark and Hex](https://medium.com/snowflake/predicting-the-unpredictable-march-madness-using-snowpark-and-hex-f16dc4f57add) by [Chase Romano](https://medium.com/@chasea.romano) and [Tyler White](https://medium.com/@btylerwhite).

## Solution Overview

Insert image here.

### Preparing the Environment

First, we will create a dedicated database and a few schemas to keep our data organized within the database.

### Data Ingestion

Next, we will download the source data from the Kaggle competition and store it in the `RAW` schema within the `MARCH_MADNESS` database.

### Feature Engineering

In this step, we will aggregate season statistics, tournament statistics, and coaching tenures to use as features for training our machine learning model.

### Model Preparation and Training

We will combine the features with the target variable and prepare the dataset for model training. Then, we will train a machine learning model using the prepared data.

### Model Inference

Finally, we will use the trained model to make predictions for the 2023 NCAA March Madness tournament.