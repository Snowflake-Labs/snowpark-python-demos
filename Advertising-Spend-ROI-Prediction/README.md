# Snowpark For Python -- Advertising Spend and ROI Prediction

This is the same demo that was presented during the [Snowflake Summit Opening Keynote](https://events.snowflake.com/summit/agenda/session/849836).

## Prereq

* Access to Snowflake account

## Setup

### **Step 1** -- Create Tables, Users, Roles

* Create table CAMPAIGN_SPEND from data hosted on publicly accessible S3 bucket

  ```sql
  CREATE or REPLACE file format csvformat
    skip_header = 1
    type = 'CSV';

  CREATE or REPLACE stage campaign_data_stage
    file_format = csvformat
    url = 's3://sfquickstarts/Summit 2022 Keynote Demo/campaign_spend/';

  CREATE or REPLACE TABLE CAMPAIGN_SPEND (
    CAMPAIGN VARCHAR(60), 
    CHANNEL VARCHAR(60),
    DATE DATE,
    TOTAL_CLICKS NUMBER(38,0),
    TOTAL_COST NUMBER(38,0),
    ADS_SERVED NUMBER(38,0)
  );

  COPY into CAMPAIGN_SPEND
    from @campaign_data_stage;
  ```

* Create table MONTHLY_REVENUE from data hosted on publicly accessible S3 bucket

  ```sql
  CREATE or REPLACE stage monthly_revenue_data_stage
    file_format = csvformat
    url = 's3://sfquickstarts/Summit 2022 Keynote Demo/monthly_revenue/';

  CREATE or REPLACE TABLE MONTHLY_REVENUE (
    YEAR NUMBER(38,0),
    MONTH NUMBER(38,0),
    REVENUE FLOAT
  );

  COPY into MONTHLY_REVENUE
    from @monthly_revenue_data_stage;
  ```

* Create table BUDGET_ALLOCATIONS_AND_ROI that holds the last six months of budget allocations and ROI

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

* Create stages required for Stored Procedures, UDFs, and saving model files

  ```sql
  CREATE OR REPLACE STAGE dash_sprocs;
  CREATE OR REPLACE STAGE dash_models;
  CREATE OR REPLACE STAGE dash_udfs;
  ```

## Notebook and Streamlit App

### **Step 1** -- Create Conda Environment

* `pip install conda`

  * ***NOTE***: The other option is to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
  
* `conda create --name snowpark -c https://repo.anaconda.com/pkgs/snowflake python=3.8`

### **Step 2** -- Install Snowpark for Python and other libraries in Conda environment

* `conda activate snowpark`

* `pip install "snowflake-snowpark-python[pandas]"`

* `pip install notebook`

* `pip install ipykernel`

* `pip install scikit-learn`

### **Step 3** -- Update [connection.json](connection.json) with your Snowflake account details and "Data Science" role credentials

### **Step 4** -- Run through the [Jupyter notebook](Snowpark_For_Python.ipynb)

In a terminal window, browse to the folder where you have this Notebook downloaded and run `jupyter notebook`

The notebook does the following...

* Performs Exploratory Data Analysis (EDA)
* Creates features for training a model and writes them to a Snowflake table
* Creates a Stored Proc for training a ML model and uploads the model to a stage
* Calls the Stored Proc to train the model
* Creates a User-Defined Function (UDF) that uses the model for inference on new data points passed in as parameters
  * NOTE: This UDF is called from the Streamlit app

### **Step 5** -- Run Streamlit app

In a terminal window, browse to this folder where you have this file downloaded and run the [Streamlit app](Snowpark_Streamlit_Revenue_Prediction.py) by executing `streamlit run Snowpark_Streamlit_Revenue_Prediction.py`

If all goes well, you should the following app in your browser window.

https://user-images.githubusercontent.com/1723932/175127637-9149b9f3-e12a-4acd-a271-4650c47d8e34.mp4
