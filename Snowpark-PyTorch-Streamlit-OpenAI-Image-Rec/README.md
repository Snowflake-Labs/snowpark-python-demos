# Image Recognition app in Snowflake using Snowpark Python, PyTorch, Streamlit and OpenAI

## Overview

In this demo, we will review how to build image recognition apps in Snowflake using Snowpark for Python, PyTorch, Streamlit and OpenAI's [DALL-E 2](https://openai.com/dall-e-2/) -- "a new AI system that can create realistic images and art from a description in natural language".

***NOTE: All necessary model files are included in this repo and I'd also like to take this moment and extend a huge thank you to the [authors](https://github.com/d-li14/mobilenetv3.pytorch#citation) for the research and making the pre-trained models available under [MIT License](https://github.com/d-li14/mobilenetv3.pytorch/blob/master/LICENSE).)***

## Prerequisites

* [Snowflake account](https://signup.snowflake.com/)
  * Login to your [Snowflake account](https://app.snowflake.com/) with the admin credentials that were created with the account in one browser tab (a role with ORGADMIN privileges). Keep this tab open during the workshop.
    * Click on the **Billing** on the left side panel
    * Click on [Terms and Billing](https://app.snowflake.com/terms-and-billing)
    * Read and accept terms to continue with the workshop
  * As ACCOUNTADMIN role
    * Create a [Warehouse](https://docs.snowflake.com/en/sql-reference/sql/create-warehouse.html), a [Database](https://docs.snowflake.com/en/sql-reference/sql/create-database.html) and a [Schema](https://docs.snowflake.com/en/sql-reference/sql/create-schema.html)
* (***Optionally***) [OpenAI](https://beta.openai.com/overview) account for the second variation of the app. Once the account is created, generate OpenAI API key to use in the application. ***Note: At the time of writing this, creating a new OpenAI account granted you $18.00 credit which is plenty for this application.***
* Your favorite IDE - Jupyter Notebook, VS Code, etc.

## Setup Environment

### **Step 1** -- Clone repository

* Clone this repo and execute `cd Snowpark-PyTorch-Streamlit-OpenAI-Image-Rec`

### **Step 2** -- Create and activate Conda environment

* Note: You can download the miniconda installer from
https://conda.io/miniconda.html. ***(OR, you may use any other Python environment with Python 3.8.)***
  
* `conda create --name snowpark -c https://repo.anaconda.com/pkgs/snowflake python=3.8`

* `conda activate snowpark`

### **Step 3** -- Install Snowpark for Python, Streamlit and other libraries in Conda environment

* `conda install -c https://repo.anaconda.com/pkgs/snowflake snowflake-snowpark-python pandas streamlit notebook cachetools`

### **Step 4** -- Create Snowflake table and internal stage

* In your Snowflake account, create a table and internal stage by running the following commands in Snowsight. ***(The table will store image data and the stage is the location for storing Snowpark Python UDF.)***

```sql
create or replace table images (file_name string, image_bytes string);

create or replace stage dash_files;
```

## Usage

### **Step 1** -- Update [connection.json](connection.json) with your Snowflake account details and credentials

* Note: For the **account** parameter, specify your [account identifier](https://docs.snowflake.com/en/user-guide/admin-account-identifier.html) and do not include the snowflakecomputing.com domain name. Snowflake automatically appends this when creating the connection.

### **Step 2** -- Create Snowpark Python User-Defined Function using PyTorch

* In a terminal window, run `jupyter notebook` at the command line from folder ***Snowpark-PyTorch-Streamlit-OpenAI-Image-Rec***.
* Open and run through the [Jupyter notebook](Snowpark_PyTorch_Image_Rec.ipynb)
  * Note: Make sure the Jupyter notebook Python kernel is set to ***snowpark***

### **Step 3** -- Run Application Variation 1 - Upload an image

* In a terminal window, execute `streamlit run Snowpark_PyTorch_Streamlit_Upload_Image_Rec.py` command from folder ***Snowpark-PyTorch-Streamlit-OpenAI-Image-Rec***.

* If all goes well, you should see the following app in your browser window.

![Image Recognition app in Snowflake using Snowpark Python, PyTorch and Streamlit](assets/app1.png "Image Recognition app in Snowflake using Snowpark Python, PyTorch and Streamlit")

### **Step 4** -- Run Application Variation 2 - OpenAI Generated image

* In a terminal window, execute `streamlit run Snowpark_PyTorch_Streamlit_OpenAI_Image_Rec.py` command folder ***Snowpark-PyTorch-Streamlit-OpenAI-Image-Rec***.

* If all goes well, you should see the following app in your browser window.

![Image Recognition app in Snowflake using Snowpark Python, PyTorch, Streamlit and OpenAI](assets/app2.png "Image Recognition app in Snowflake using Snowpark Python, PyTorch, Streamlit and OpenAI")

---

For questions and feedback, please reach out to [Dash](https://twitter.com/iamontheinet).
