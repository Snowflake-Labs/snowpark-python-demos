# Overview:
This is an example of using Linear Regression model in Scikit-Learn, Snowpark and Python UDFs to predict customer spend. The training of the model happens locally while the scoring is done within Snowflake using the UDF created via Snowpark in the Jupyter Notebook. The sample data file is EcommerceCustomers. Model training is done on a test dataset while scoring is done on the entire dataset. 


An ecommerce retailer is looking to use machine learning to understand its customer's online engagement with its digital outlets i.e website and app. It is trying to decide whether to focus its efforts on the mobile app experience or website. We will use Linear Regression model to see which user acitivity has the biggest impact on their likelyhood of spending more money.

Variables of interest:

Avg. Session Length: Average session of in-store style advice sessions.
Time on App: Average time spent on App in minutes
Time on Website: Average time spent on Website in minutes
Length of Membership: How many years the customer has been a member 

# Prerequisites:
Snowpark for Python library v.06

* Snowflake account
* Snowpark for Python
* The examples also use the following Python libraries:
   ```
   scikit-learn
   pandas
   numpy
   matplotlib
   seaborn
   streamlit
   ```
* Jupyter or JupyterLab
If any of the packages used in the example are not part of your python environment, you can install them using
<br>`import sys`<br>
`!conda install --yes --prefix {sys.prefix} <package_name>`
* Latest streamlit package, which you can get by
 `!pip install streamlit`

## What You'll Learn:
Simple introductory tutorial on Snowpark. Covers data ingestion, data science and creation of an app using Streamlit

* How to use Snowpark for Python for doing Feature Engineering
* Training a Machine Learning model outside of Snowflake and to deploy it as a Python UDF
* Visualizing your working model in an end user app

    
# Usage/Steps

1. Open terminal and clone this repo or use GitHub Desktop, since it is part of the snowflakecorp organisation you need to set up the authentification before cloning: 

    `git clone https://github.com/Snowflake-Labs/snowpark-python-demos`

2. Change to the `Predict Customer Spend` directory and launch  JupyterLab

    `jupyter lab`

3. Paste the URL in a browser window and once JupyterLab comes up, switch to the work directory and update `creds_generic.json` and rename it to `creds.json` to reflect your snowflake environment.

4. To run streamlit (ecommapp), on your terminal run  `streamlit run ecommapp.py`
   Here's what the app will look like:
   
![ecommapp](https://user-images.githubusercontent.com/1723932/179316941-87b298f2-43de-4635-a0b1-bdc68f059605.png)
