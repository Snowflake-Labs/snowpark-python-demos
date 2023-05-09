
# Advertising Spend and ROI Prediction

This is the same demo that was presented during the [Snowflake Summit Opening Keynote](https://events.snowflake.com/summit/agenda/session/849836). It is built using Snowpark For Python and Streamlit. For questions and feedback, please reach out to <dash.desai@snowflake.com>.

## Overview

In this workshop, we will train a Linear Regression model to predict future ROI (Return On Investment) of variable advertising spend budgets across multiple channels including search, video, social media, and email using Snowpark for Python and scikit-learn. By the end of the session, you will have an interactive web application deployed visualizing the ROI of different allocated advertising spend budgets. NOTE: The accompanying slides can be found [here](https://github.com/Snowflake-Labs/snowpark-python-demos/blob/77f54633f850c66053dfa055c82a7fc6dec8deca/Advertising-Spend-ROI-Prediction/Snowpark%20for%20Python%20And%20Streamlit%20ML%20Workshop.pdf).

Workshop highlights:

* Set up your favorite IDE (e.g. Jupyter, VSCode) for Snowpark and ML
* Analyze data and perform data engineering tasks using Snowpark DataFrames
* Use open-source Python libraries from a curated Snowflake Anaconda channel with near-zero maintenance or overhead
* Deploy ML model training code on Snowflake using Python Stored Procedure
* Create and register Scalar and Vectorized Python User-Defined Functions (UDFs) for inference
* Create Snowflake Task to automate (re)training of the model
* Create Streamlit web application that uses the Scalar UDF for real-time inference on new data points based on user input

If all goes well, you should see the following app in your browser window.

https://user-images.githubusercontent.com/1723932/175127637-9149b9f3-e12a-4acd-a271-4650c47d8e34.mp4

## Prerequisites

* [Snowflake account](https://signup.snowflake.com/)
  * Login to your [Snowflake account](https://app.snowflake.com/) with the admin credentials that were created with the account in one browser tab (a role with ORGADMIN privileges). Keep this tab open during the workshop.
    * Click on the **Billing** on the left side panel
    * Click on [Terms and Billing](https://app.snowflake.com/terms-and-billing)
    * Read and accept terms to continue with the workshop
  * As SYSADMIN role
    * Create a [Warehouse](https://docs.snowflake.com/en/sql-reference/sql/create-warehouse.html), a [Database](https://docs.snowflake.com/en/sql-reference/sql/create-database.html) and a [Schema](https://docs.snowflake.com/en/sql-reference/sql/create-schema.html)

## Setup

---
```sql
  USE ROLE ACCOUNTADMIN;
```
---
### **Step 1** -- Create Tables, Load Data and Setup Stages

An example database, tables, stages, and file formats will be configured and created as you're executing the (Snowpark_For_Python.ipynb) notebook.

These stages will be configured to use data hosted on a publicly accessible S3 bucket.

For the Streamlit application, you will need to create a table BUDGET_ALLOCATIONS_AND_ROI that holds the last six months of budget allocations and ROI.

---
```sql
  CREATE or REPLACE TABLE BUDGET_ALLOCATIONS_AND_ROI (
    MONTH varchar(30),
    SEARCHENGINE integer,
    SOCIALMEDIA integer,
    VIDEO integer,
    EMAIL integer,
    ROI float
  );

  INSERT INTO BUDGET_ALLOCATIONS_AND_ROI (MONTH, SEARCHENGINE, SOCIALMEDIA, VIDEO, EMAIL, ROI)
  VALUES
  ('January',35,50,35,85,8.22),
  ('February',75,50,35,85,13.90),
  ('March',15,50,35,15,7.34),
  ('April',25,80,40,90,13.23),
  ('May',95,95,10,95,6.246),
  ('June',35,50,35,85,8.22);
```
---
## Notebook and Streamlit App

### **Step 1** -- Clone Repo

* `git clone https://github.com/Snowflake-Labs/snowpark-python-demos` OR `git clone git@github.com:Snowflake-Labs/snowpark-python-demos.git`

* `cd Advertising-Spend-ROI-Prediction`

### **Step 2** -- Create And Activate Conda Environment

* Note: You can download the miniconda installer from
https://conda.io/miniconda.html. OR, you may use any other Python environment with Python 3.8

* `conda create --name snowpark -c https://repo.anaconda.com/pkgs/snowflake python=3.8`

* `conda activate snowpark`

### **Step 3** -- Install Snowpark for Python, Streamlit and other libraries in Conda environment

* `conda install -c https://repo.anaconda.com/pkgs/snowflake snowflake-snowpark-python pandas notebook scikit-learn cachetools streamlit`

### **Step 4** -- Update [connection.json](connection.json) with your Snowflake account details and credentials

* Note: For the **account** parameter, specify your [account identifier](https://docs.snowflake.com/en/user-guide/admin-account-identifier.html) and do not include the snowflakecomputing.com domain name. Snowflake automatically appends this when creating the connection.

### **Step 5** -- Train & deploy ML model

* In a terminal window, browse to the folder where you have this Notebook downloaded and run `jupyter notebook` at the command line
* Open and run through the [Jupyter notebook](Snowpark_For_Python.ipynb)
  * Note: Make sure the Jupyter notebook (Python) kernel is set to ***snowpark***

The notebook does the following...

* Performs Exploratory Data Analysis (EDA)
* Creates features for training a model and writes them to a Snowflake table
* Creates a Stored Proc for training a ML model and uploads the model to a Snowflake stage
* Calls the Stored Proc to train the model
* Creates Scalar and Vectorized User-Defined Functions (UDFs) that use the model for inference on new data points passed in as parameters
  * Note: The Scalar UDF is called from the below Streamlit app for real-time inference on new budget allocations based on user input
* Creates a Snowflake Task to automate (re)training of the model

### **Step 6** -- Run Streamlit app

* In a terminal window, browse to this folder where you have this file downloaded and run the [Streamlit app](Snowpark_Streamlit_Revenue_Prediction.py) by executing `streamlit run Snowpark_Streamlit_Revenue_Prediction.py`

* If all goes well, you should see the following app in your browser window.

https://user-images.githubusercontent.com/1723932/175127637-9149b9f3-e12a-4acd-a271-4650c47d8e34.mp4
