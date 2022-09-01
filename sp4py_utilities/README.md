# sp4py_utilities

## 1. Overview 
Example of how scikit-learn preprocessing functionality can be implemented using Snowpark for Python, enabling to do most of the preprocessing of data within Snowflake.

## 2. Prerequisite
* Snowflake account
* Snowpark for Python
* The modules are depended on the following Python libraries:
   ```
   scipy
   numpy
   ```

## 3. What you'll learn
This example shows how Snowpark for Python can be extended with similar functionality as scikit-learn preprocessing. 

## 4. Usage/Steps
### Preprocessing
A module for data preprocessing of numeric and categorical features/columns using Snowpark DataFrames.

The functions should in most cases follow the [sklearn.preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) 
library with fit, transform etc methods and mostly the same parameters.

If the fitted scaler/encoder is to be used with a Python UDF then the **udf_transform** module needs to be used for the 
transformation.

In order to use the module download the **preprocessing** folder and make sure it is visible to your python environment.

For more examples see the [**preprocessing_demo** notebook](preprocessing_demo.ipynb).

#### Scalers
The preprocessing module has the following scalers that can be used with numeric features:
* MinMaxScaler: Transform each column by scaling each feature to a given range.
* StandardScaler: Standardize features by removing the mean and scaling to unit variance.
* MaxAbsScaler: Scale each column by its maximum absolute value.
* RobustScaler: Scale features using statistics that are robust to outliers.
* Normalizer: Normalize individually to unit norm.
* Binarizer: Binarize data (set feature values to 0 or 1) according to a threshold.

Example using the MinMaxScaler
```
import preprocessing as pp

session = Session.builder.configs(connection_parameters).create()
df_housing = session.table("california_housing")

# The columns to scale
input_cols=["MedInc", "AveOccup"]
# The columns that holds the scaled values
output_cols = ["MedInc_scaled", "AveOccup_scaled"]

# Create a MinMax Scaler object that will scale the input_cols and return scaled values using output_cols
mms = pp.MinMaxScaler(input_cols=input_cols, output_cols=output_cols)
# Fit the values needed for the scaler for each of the input_cols 
mms.fit(df_housing)
# Scale the input_cols and return the scaled values in the output_cols in the returned Snowpark DataFrame, the actual
# scaling are not done until a action method is called ie show(), collect() etc.
mms_tr_df = mms.transform(df_housing)
```

#### Encoders
The preprocessing module has the following encoders that can be used with categorical features:
* OneHotEncoder: Encode categorical features as a one-hot.
* OrdinalEncoder: Encodes a string column of labels to a column of label indices. The indices are in [0, number of labels].
* LabelEncoder: A label indexer that maps a string column of labels to a column of label indices.

Example using the OneHotEncoder
```
import preprocessing as pp

session = Session.builder.configs(connection_parameters).create()
df_churn = session.table("CUSTOMER_CHURN")

encoder_input_cols = ["STATE", "AREA_CODE", "INTL_PLAN"]

# Create a One hot encoder object that will encode the input_cols
ohe = pp.OneHotEncoder(input_cols=encoder_input_cols)
# Fit the values needed for the encoder for each of the input_cols
ohe.fit(df_churn)
# Encode the input_cols and return the scaled values in the output_cols in the returned Snowpark DataFrame, the actual
# encoding are not done until a action method is called ie show(), collect() etc.
ohe_tr_df = ohe.transform(df_churn)
```
### UDF Transform
A module for using the fitted scalers/encoders created with the preprocessing module in Python UDFs.

In order to use the module download the **udf_transform** folder and make sure it is visible to your python environment 
and if using it for Python UDFs you also need to upload it to Snowflake, the simplest way is by using **add_imports**

For more examples see the [**udf_transform_demo** notebook](udf_transform_demo.ipynb).

The module has the following functions:
* udf_minmax_transform
* udf_minmax_inverse_transform
* udf_standard_transform
* udf_standard_inverse_transform
* udf_maxabs_transform
* udf_maxabs_inverse_transform
* udf_robust_transform
* udf_robust_inverse_transform
* udf_normalizer_transform
* udf_binarizer_transform

Input data can be a list or a numpy array.

Example using the udf_minmax_transform function:
```
import preprocessing as pp
import udf_transform as ut

session = Session.builder.configs(connection_parameters).create()

df_housing = session.table("california_housing")
input_cols=["MedInc", "AveOccup"]

# The udf_transform functions requires that data to be transformed is a  list of lists
data = [[8.3252, 2.5555555555555554], [3.6591, 2.1284046692607004]]

# First create a MinMax scaler object that will use the input_cols for fitting
mms = pp.MinMaxScaler(input_cols=input_cols)
# Fit the scaler on the input_cols  using a Snowpark Dataframe
mms.fit(df_housing)
# Get the scaler as a Dictonary object to be used with the udf_minmax_transform function
mms_udf = mms.get_udf_encoder()

# Scale the input data (list of list) using the scaler dictornary
# This can be done in a Python UDF, see the udf_transform_demo notebook for an example.
mms_scaled_data = ut.udf_minmax_transform(data, mms_udf)
```
