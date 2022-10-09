# %%
import streamlit as st
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import *
import json

# %%
# Create a session to Snowflake with credentials
with open("creds.json") as f:
    connection_parameters = json.load(f)
session = Session.builder.configs(connection_parameters).create()

# %%
# Header
head1, head2 = st.columns([8, 1])

with head1:
    st.header("Customer Spend Prediction Model")
with head2:
    st.markdown(
        f' <img src="https://api.nuget.org/v3-flatcontainer/snowflake.data/0.1.0/icon" width="50" height="50"> ',
        unsafe_allow_html=True)

st.markdown('##')
st.markdown('##')

# %%
# Customer Spend Slider Column
col1, col2, col3 = st.columns([4, 1, 10])

customer_df = session.table('PREDICTED_CUSTOMER_SPEND')

# Read Data
minasl, maxasl, mintoa, maxtoa, mintow, maxtow, minlom, maxlom = customer_df.select(
    floor(min(col("Avg. Session Length"))),
    ceil(max(col("Avg. Session Length"))),
    floor(min(col("Time on App"))),
    ceil(max(col("Time on App"))),
    floor(min(col("Time on Website"))),
    ceil(max(col("Time on Website"))),
    floor(min(col("Length of Membership"))),
    ceil(max(col("Length of Membership")))
).toPandas().iloc[0, ]

minasl = int(minasl)
maxasl = int(maxasl)
mintoa = int(mintoa)
maxtoa = int(maxtoa)
mintow = int(mintow)
maxtow = int(maxtow)
minlom = int(minlom)
maxlom = int(maxlom)

# Column 1
with col1:
    st.markdown("#### Search Criteria")
    st.markdown('##')
    asl = st.slider("Session Length", minasl, maxasl, (minasl, minasl+5), 1)
    #st.write("Session Length ", asl)
    toa = st.slider("Time on App", mintoa, maxtoa, (mintoa, mintoa+5), 1)
    #st.write("Time on App ", toa)
    tow = st.slider("Time on Website", mintow, maxtow, (mintow, mintow+5), 1)
    #st.write("Time on Website ", tow)
    lom = st.slider("Length of Membership", minlom,
                    maxlom, (minlom, minlom+4), 1)
    #st.write("Length of Membership ", lom)

# Column 2 (3)
with col3:
    #avg_sess_len = st.slider("Avg. Session Length", min_sess_len, max_sess_len, (min_sess_len,min_sess_len+1), 1)
    st.markdown("#### Customer Predicted Spend")
    st.markdown('##')

    minspend, maxspend = customer_df.filter(
        (col("Avg. Session Length") <= asl[1]) & (
            col("Avg. Session Length") > asl[0])
        & (col("Time on App") <= toa[1]) & (col("Time on App") > toa[0])
        & (col("Time on Website") <= tow[1]) & (col("Time on Website") > tow[0])
        & (col("Length of Membership") <= lom[1]) & (col("Length of Membership") > lom[0])
    ).select(trunc(min(col('PREDICTED_SPEND'))), trunc(max(col('PREDICTED_SPEND')))).toPandas().iloc[0, ]

    st.write(f'This customer is likely to spend between ')
    st.metric(label="", value=f"${minspend}")
    #st.write("and")
    st.metric(label="and", value=f"${maxspend}")

    st.markdown("---")
    st.write("\nThe biggest drivers of customer spend are:")
    st.markdown('* **Length of Membership** \n * **Time on App**')
    st.write("You can see spend range change much more when one of these two variables is changed.")
