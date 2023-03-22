import streamlit as st
import glob
import os


def main():

    # Header.
    #st.image("../logo-sno-blue.png", width=100)
    #st.subheader("Sentiment Analysis")



    # Display README.md contents.
    with open("../README.md", 'r') as f:
        readme_line = f.readlines()
        readme_buffer = []
        # resource_files
    for line in readme_line:
        readme_buffer.append(line)
    
    st.markdown(''.join(readme_buffer))

    # with open("README.md", 'r') as f:
    #     st.markdown(f.read(), unsafe_allow_html=True)

#     c1, c2, c3 = st.columns([0.2, 0.6, 0.2])
#     with c2:
#         st.image("images/DICOM_Image_Detect_Pneumonia.png")
#     st.subheader("Summary")
#     st.markdown("""In this use case to detect pneumonia, we have read X Ray images as input, trained a tensorflow model as a stored procedure in snowflake using these vectorized images and finally defined a User Defined function (UDF) from the model trained to detect the probability of Pneumonia.
#
# Medical Images are unstructured data types and snowflake allows you to vectorize these images by leveraging image reading libraries such as sci-kit image and openCV on snowpark using python. Tensorflow is also a supported library in snowpark and we can train machine learning models by pushing down the training workload to snowflake compute power to generate a model file. Thereafter using the trained tensorflow model file we can create a user defined function for inference that can detect Pneumonia by uploading a new X Ray image to get the probability of pneumonia. The streamlit library is used to build the app that works seamlessly with snowpark to perform the outlined steps to detect pneumonia.
# """)


if __name__ == '__main__':
    main()