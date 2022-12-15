# Snowpark for Python Developer Guide: https://docs.snowflake.com/en/developer-guide/snowpark/python/index.html
# Streamlit docs: https://docs.streamlit.io/
# OpenAI: https://openai.com/

import json
import pandas as pd
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import col
import streamlit as st
import io
from io import StringIO
import base64
import uuid

# Streamlit config
st.set_page_config(
    page_title="Image Recognition App in Snowflake",
    layout='wide',
    menu_items={
         'Get Help': 'https://developers.snowflake.com',
         'About': "The source code for this application can be accessed on GitHub https://github.com/iamontheinet/snowpark-pytorch-streamlit-image-rec"
     }
)

# Set page title, header and links to docs
st.header("Image Recognition app in Snowflake using Snowpark Python, PyTorch and Streamlit")
st.caption(f"App developed by [Dash](https://twitter.com/iamontheinet)")
st.write("[Resources: [Snowpark for Python Developer Guide](https://docs.snowflake.com/en/developer-guide/snowpark/python/index.html)   |   [Streamlit](https://docs.streamlit.io/)   |   [PyTorch Implementation of MobileNet V3](https://github.com/d-li14/mobilenetv3.pytorch)]")

# Function to create new or get existing Snowpark session
def create_session():
    if "snowpark_session" not in st.session_state:
        session = Session.builder.configs(json.load(open("connection.json"))).create()
        st.session_state['snowpark_session'] = session
    else:
        session = st.session_state['snowpark_session']
    return session

# Call function to create new or get existing Snowpark session to connect to Snowflake
session = create_session()

uploaded_file = st.file_uploader("Choose an image file", accept_multiple_files=False, label_visibility='hidden')
if uploaded_file is not None:

    with st.spinner("Uploading image and generating prediction in real-time..."):

        # Convert image base64 string into hex 
        bytes_data_in_hex = uploaded_file.getvalue().hex()

        # Generate new image file name
        file_name = 'img_' + str(uuid.uuid4())

        # Write image data in Snowflake table
        df = pd.DataFrame({"FILE_NAME": [file_name], "IMAGE_BYTES": [bytes_data_in_hex]})
        session.write_pandas(df, "IMAGES")

        # Call Snowpark User-Defined Function to predict image label
        predicted_label = session.sql(f"SELECT image_recognition_using_bytes(image_bytes) as PREDICTED_LABEL from IMAGES where FILE_NAME = '{file_name}'").to_pandas().iloc[0,0]

        _, col2, col3, _ = st.columns(4, gap='medium')
        with st.container():
            with col2:
                # Display uploaded image
                st.subheader("Image you uploaded")
                st.image(uploaded_file)

            with col3:
                # Display predicted label
                st.subheader("Predicted Label in Snowpark using PyTorch")
                st.code(predicted_label, language='python')
