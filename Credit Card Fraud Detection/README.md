# Credit Card Fraud Detection Demo

## 1. Overview
This example is based on the Machine Learning for Credit Card Fraud detection - Practical handbook, https://fraud-detection-handbook.github.io/fraud-detection-handbook/

It shows how to do Feature Enginerring with Snowpark, preparing data for training a Machin Leraning model and finaly how to deploy and use a trained model in Snowflake using Python UDF.


## 2. Prerequisite

* Snowflake account
* Snowpark for Python
* The examples also use the following Python libraries:
   ```
   scikit-learn
   pandas
   numpy
   matplotlib
   ```
* Jupyter or JupyterLab

## 3. What you'll learn
* How to use Snowpark for Python for doing Feature Engineering
* How you can create a custom sampling functioand with Snowpark for Python
* Training a Machine Learning model outside of Snowflake nd to deploy it as a Python UDF

## 4. Usage/Steps

1. Open terminal and clone this repo or use GitHub Desktop, since it is part of the snowflakecorp organisation you need to set up the authentification before cloning: 

    `git clone https://github.com/Snowflake-Labs/snowpark-python-examples`

2. Change to the `Credit Card Fraud Detection` directory and launch  JupyterLab

    `jupyter lab`

6. Paste the URL in a browser window and once JupyterLab comes up, switch to the work directory and update `creds.json` to reflect your snowflake environment.

In order to load data you can either run the `00 - Snowpark Python - Load Data.ipynb` notebook.

Or "manual" load it by following the steps below
1. In your snowflake account create the following table:

```
create or replace TABLE CUSTOMER_TRANSACTIONS_FRAUD (
 TRANSACTION_ID NUMBER,  
 TX_DATETIME TIMESTAMP_NTZ, 
 CUSTOMER_ID NUMBER, 
 TERMINAL_ID NUMBER, 
 TX_AMOUNT FLOAT, 
 TX_TIME_SECONDS NUMBER, 
 TX_TIME_DAYS NUMBER, 
 TX_FRAUD NUMBER, 
 TX_FRAUD_SCENARIO NUMBER);
```

3. Load the data/fraud_transactions.csv.gz into CUSTOMER_TRANSACTIONS_FRAUD

4. After loading the CUSTOMER_TRANSACTIONS_FRAUD table generate the CUSTOMERS and TERMINALS tables using the following SQL

```
CREATE TABLE CUSTOMERS
AS
 SELECT DISTINCT CUSTOMER_ID FROM CUSTOMER_TRANSACTIONS_FRAUD
  ORDER BY CUSTOMER_ID;
```

```
 CREATE TABLE TERMINALS
 AS
 SELECT DISTINCT TERMINAL_ID FROM CUSTOMER_TRANSACTIONS_FRAUD
  ORDER BY TERMINAL_ID;
```