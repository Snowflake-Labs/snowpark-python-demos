# RETAIL CHURN PREDICTION 

## 1. Overview
We will be working with three tables that contains Customer data, Communications data and Order data for a Retail Store. These dataset can be used to understand the consumer behaviour to predict churn, in this example.

These datasets were generated for this demo using a Kaggle dataset below.

Reference: https://www.kaggle.com/uttamp/store-data

In this demo, we will be leveraging Python Stored Procedures.

## 2. Prerequisite

* Snowflake account

* Snowpark for Python in your local machine

Please follow the instructions to install Snowpark: https://docs.snowflake.com/en/developer-guide/snowpark/python/setup.html#installation-instructions

* The examples also use the following Python libraries:
   ```
   json
   scipy
   joblib
   sklearn
   cachetools
   pandas
   numpy
   matplotlib
   xgboost
  
   ```
This code also leverages functions defined in the sp4py repo: https://github.com/Snowflake-Labs/snowpark-python-examples/tree/main/sp4py_utilities

* Jupyter or JupyterLab
* Upload the 3 CSV files in /data into an external stage (e.g S3). In the object store (e.g. S3 bucket), create 3 directories /customer, /order, /comm_hist and upload the source files src_customer.csv, src_order.csv, src_comm_hist.csv respectively to each of these 3 directories. 

* Create a storage integration and a stage (e.g. churn_source_data) using the documentation (https://docs.snowflake.com/en/sql-reference/sql/create-storage-integration.html)  and create three external tables as below:

```
CREATE STAGE CHURN_SOURCE_DATA 
storage_integration = <your_storage_integration>
  url = 's3://<your bucket>'
  file_format = (type=CSV);
 ```


```
CREATE OR REPLACE EXTERNAL TABLE SRC_CUSTOMER 
(CUSTOMER_ID VARCHAR(40) as (value:c1::varchar), 
 CREATED_DT DATE as (value:c2::date), 
 CITY VARCHAR(40) as (value:c3::varchar), 
 STATE VARCHAR(2) as (value:c4::varchar), 
 FAV_DELIVERY_DAY VARCHAR(40) as (value:c5::varchar), 
 REFILL  NUMBER(38,0) as (value:c6::integer), 
 DOOR_DELIVERY  NUMBER(38,0) as (value:c7::integer), 
 PAPERLESS  NUMBER(38,0) as (value:c8::integer), 
 CUSTOMER_NAME VARCHAR(40) as (value:c9::varchar),  
 RETAINED  NUMBER(38,0) as (value:c10::integer) 
) 
  LOCATION = @churn_source_data/customer/ 
  REFRESH_ON_CREATE = TRUE 
  AUTO_REFRESH = TRUE 
  FILE_FORMAT = ( TYPE = CSV SKIP_HEADER=1)
  ```
```
CREATE OR REPLACE EXTERNAL TABLE SRC_ORDER 
  (CUSTOMER_ID VARCHAR(40) as (value:c1::varchar), 
   ORDER_DT VARCHAR(40) as (value:c2::varchar), 
   CITY VARCHAR(40) as (value:c3::varchar), 
   STATE VARCHAR(2) as (value:c4::varchar), 
   ORDER_AMOUNT FLOAT  as (value:c5::float), 
   ORDER_ID VARCHAR(40)  as (value:c6::varchar) 
) 
  LOCATION = @churn_source_data/order/ 
  REFRESH_ON_CREATE = TRUE 
  AUTO_REFRESH = TRUE 
  FILE_FORMAT = ( TYPE =  CSV SKIP_HEADER=1) 
 ``` 
 ```
CREATE OR REPLACE EXTERNAL TABLE SRC_COMMUNICATION_HIST 
 ( CUSTOMER_ID VARCHAR(40) as (value:c1::varchar), 
   ESENT NUMBER(38,0) as (value:c2::integer), 
   EOPENRATE FLOAT as (value:c3::float), 
   ECLICKRATE FLOAT as (value:c4::float) 
)
  LOCATION = @churn_source_data/comm_hist 
  REFRESH_ON_CREATE = TRUE 
  AUTO_REFRESH = TRUE 
  FILE_FORMAT = ( TYPE =  CSV SKIP_HEADER=1) 
  ```
  
  We will use these external tables to read the data, do data engineering.

## 3. What you'll learn

* How to use Snowpark for Python for doing Feature Engineering
* How you can apply transformations with Snowpark for Python
* Training a Machine Learning model and creating predictions table in Snowflake using Python Stored Procedures and Python UDFs. 

## 4. Usage/Steps

1. Open terminal and clone this repo or use GitHub Desktop, since it is part of the snowflakecorp organisation you need to set up the authentification before cloning: 

    `git clone https://github.com/Snowflake-Labs/snowpark-python-examples`

2. Change to the `Retail-Churn-Analytics` directory and launch  JupyterLab or Jupyter

    `jupyter lab`
    
    `jupyter notebook`

3. Paste the URL in a browser window and switch to the work directory and update `creds.json` to reflect your snowflake environment.
4. Start running the notebooks, DE01, DS02.

 



