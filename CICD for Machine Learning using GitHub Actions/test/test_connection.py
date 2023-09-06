import sys
import os
import json
from datetime import datetime
import re

from snowflake.snowpark.session import Session
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T
from snowflake.snowpark.functions import col

from snowflake.ml.modeling.preprocessing import OneHotEncoder

import configparser
import os

# Create a ConfigParser object
config = configparser.ConfigParser()

# Define the path to the config file
config_path = os.path.expanduser("~/.snowsql/config")

# Read the contents of the file into the ConfigParser object
config.read(config_path)

# Access the values using the section and key
# Assuming the values you want are in the "connections.dev" section
dict_creds = {}

dict_creds['account'] = config['connections.dev']['accountname']
dict_creds['user'] = config['connections.dev']['username']
dict_creds['password'] = config['connections.dev']['password']
dict_creds['role'] = config['connections.dev']['rolename']
dict_creds['database'] = config['connections.dev']['dbname']
dict_creds['warehouse'] = config['connections.dev']['warehousename']

print("\n\nCreds")
print(dict_creds)

# Testing Snowflake Connection
session = Session.builder.configs(dict_creds).create()
session.use_database('ML_SNOWPARK_CI_CD')
session.use_schema('DATA_PROCESSING')

# Creating a Snowpark DataFrame
application_record_sdf = session.table('APPLICATION_RECORD')
credit_record_sdf = session.table('CREDIT_RECORD')
print('\n\nApplication table size\t: ',application_record_sdf.count(), 
      '\nCredit table size\t: ', credit_record_sdf.count())