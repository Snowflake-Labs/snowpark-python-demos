import streamlit as st
from streamlit_option_menu import option_menu

from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import udf, col
from snowflake.snowpark.types import Variant
from snowflake.snowpark import functions as fn

import pandas as pd
import numpy as np


from snowflake.snowpark import version
print(f"Snowflake snowpark version is : {version.VERSION}")

# create_snowflake_connexion
def create_snowflake_connexion():
    import json

    if "snowpark_session" not in st.session_state:
        session = Session.builder.configs(json.load(open("../connection.json"))).create()
        print(session)

        session.clear_imports()
        session.clear_packages()
        session.add_packages("snowflake-snowpark-python")
        session.add_packages("scikit-learn", "pandas", "numpy", "nltk", "joblib", "cachetools")

        st.session_state['snowpark_session'] = session
    else:
        session = st.session_state['snowpark_session']
    return session

def create_snowflake_connexion_tmp():
    from config import snowflake_conn_prop
    session = Session.builder.configs(snowflake_conn_prop).create()

    print(session.sql('use role ACCOUNTADMIN').collect())
    #print(session.sql('use IMDB.PUBLIC').collect())
    #print(session.sql('select current_role(), current_warehouse(), current_database(), current_schema()').collect())

    session.add_packages("snowflake-snowpark-python")
    session.add_packages("scikit-learn", "pandas", "numpy", "nltk", "joblib", "cachetools")

    return session


##################################################################
# ML
##################################################################

# #### Train function
def train_model_review_pipline(session : Session, train_dataset_name: str) -> Variant:
    
    from nltk.corpus import stopwords
    import sklearn.feature_extraction.text as txt
    from sklearn import svm
    import os
    from joblib import dump
        
    train_dataset = session.table(train_dataset_name)
    train_dataset_flag = train_dataset.withColumn("SENTIMENT_FLAG", fn.when(train_dataset.SENTIMENT == "positive", 1)
                                     .otherwise(2))
    train_x = train_dataset_flag.toPandas().REVIEW.values
    train_y = train_dataset_flag.toPandas().SENTIMENT_FLAG.values
    print('Taille train x : ', len(train_x))
    print('Taille train y : ', len(train_y))
    
    print('Configuring parameters ...')
    # bags of words: parametrage
    analyzer = u'word' # {‘word’, ‘char’, ‘char_wb’}
    ngram_range = (1,2) # unigrammes
    token = u"[\\w']+\\w\\b" #
    max_df=0.02    #50. * 1./len(train_x)  #default
    min_df=1 * 1./len(train_x) # on enleve les mots qui apparaissent moins de 1 fois
    binary=True # presence coding
    svm_max_iter = 100
    svm_c = 1.8
    
    print('Building Sparse Matrix ...')
    vec = txt.CountVectorizer(
        token_pattern=token, \
        ngram_range=ngram_range, \
        analyzer=analyzer,\
        max_df=max_df, \
        min_df=min_df, \
        vocabulary=None, 
        binary=binary)

    # pres => normalisation
    bow = vec.fit_transform(train_x)
    print('Taille vocabulaire : ', len(vec.get_feature_names_out()))
    
    print('Fitting model ...')
    model = svm.LinearSVC(C=svm_c, max_iter=svm_max_iter)
    print(model.fit(bow, train_y))
    
    # #### Create a stage to store the model
    session.sql("CREATE STAGE IF NOT EXISTS MODELS").collect()
    
    # Upload the Vectorizer (BOW) to a stage
    print('Upload the Vectorizer (BOW) to a stage')
    model_output_dire = '/tmp'
    model_file = os.path.join(model_output_dire, 'vect_review.joblib')
    dump(vec, model_file, compress=True)
    session.file.put(model_file, "@MODELS", auto_compress=False, overwrite=True)
    
    # Upload trained model to a stage
    print('Upload trained model to a stage')
    model_output_dire = '/tmp'
    model_file = os.path.join(model_output_dire, 'model_review.joblib')
    dump(model, model_file, compress=True)
    session.file.put(model_file, "@MODELS", auto_compress=False, overwrite=True)
    
    return {"STATUS": "SUCCESS", "R2 Score Train": str(model.score(bow, train_y))}


# Function to load the model from the Internal Stage (Snowflake)
import cachetools
@cachetools.cached(cache={})
def load_file(filename):
    
    import joblib
    import sys
    import os
    
    import_dir = sys._xoptions.get("snowflake_import_directory")
    
    if import_dir:
        with open(os.path.join(import_dir, filename), 'rb') as file:
            m = joblib.load(file)
            return m


##################################################################
# FUNCTIONS
##################################################################

st.set_page_config(
        page_title="Snowflake Sentiment Analysis Demo",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Header.
st.image("../logo-sno-blue.png", width=100)
st.subheader("Sentiment Analysis")
st.markdown('----')

#st.set_page_config(page_title="Sentiment Analysis",page_icon="❄️")
#st.title('Sentiment Analysis APP')


# Add header and a subheader
#st.header("Sentiment Analysis")

schema_name = "PUBLIC"
target_column_name = "SENTIMENT"
train_function_name = "train_model_review_pipline"

def train_model (session, table, cwh, cwh_size, use_optimized):

    if (use_optimized):
        cmd = "alter warehouse " + cwh + " suspend"
        session.sql(cmd).collect()
    
        cmd = "alter warehouse " + cwh + " set warehouse_size = '2X-LARGE'"
        session.sql(cmd).collect()
    
        cmd = "alter warehouse " + cwh + " set WAREHOUSE_TYPE = 'SNOWPARK-OPTIMIZED'"
        session.sql(cmd).collect()
    
    st.write("Register a Store Procedure : train_model_review_pipline")
    session.sproc.register(func=train_model_review_pipline, name=train_function_name, replace=True)
    st.write("Call the Store Procedure : train_model_review_pipline")
    session.call(train_function_name, table)
    
    # Import the needed files from the stage
    st.write("Import models : vect_review and model_review")
    session.add_import("@MODELS/model_review.joblib")
    session.add_import("@MODELS/vect_review.joblib")

    # Deploy an UDF for prediction
    st.write("Deploy an UDF called : predict_review for Inference (Prediction)")
    @udf(name='predict_review', session=session, is_permanent = False, stage_location = '@MODELS', replace=True)
    def predict_review(args: list) -> float:
        
        import sys
        import pandas as pd
        from joblib import load

        model = load_file("model_review.joblib")
        vec = load_file("vect_review.joblib")
            
        features = list(["REVIEW", "SENTIMENT_FLAG"])
        
        row = pd.DataFrame([args], columns=features)
        bowTest = vec.transform(row.REVIEW.values)
        
        return model.predict(bowTest)

    if (use_optimized):
        cmd = "alter warehouse " + cwh + " suspend"
        session.sql(cmd).collect()
    
        cmd = "alter warehouse " + cwh + " set WAREHOUSE_TYPE = 'STANDARD'"
        session.sql(cmd).collect()

        cmd = "alter warehouse " + cwh + " set warehouse_size = '" + cwh_size + "'"
        session.sql(cmd).collect()

def assess_performance(y_pred, y_test):
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    Accuracy = accuracy_score(y_test, y_pred) * 100
    Recall = recall_score(y_test, y_pred, average='weighted') * 100
    Precision = precision_score(y_test, y_pred, average='weighted') * 100
    F1_Score = f1_score(y_test, y_pred, average='weighted') * 100
    return Accuracy, Recall, Precision, F1_Score

def assess_performance_df(y_pred, y_test):

    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred, average='weighted'))
    print("Precision: ", precision_score(y_test, y_pred, average='weighted'))
    print("F1 Score: ", f1_score(y_test, y_pred, average='weighted'))
    d = {'Metric': ['Accuracy', 'Recall', 'Precision', 'F1 Score'], 'Score (%)': [accuracy_score(y_test, y_pred) * 100, \
                                                                            recall_score(y_test, y_pred, average='weighted') * 100, \
                                                                            precision_score(y_test, y_pred, average='weighted') * 100, \
                                                                            f1_score(y_test, y_pred, average='weighted') * 100]}
    df = pd.DataFrame(data=d)
    return df


##################################################################
# STREAMLIT APP
##################################################################

# page custom Styles
st.markdown("""
            <style>

            
                div.stButton > button:first-child {
                    background-color: #50C878;color:white; border-color: none;
                }
            </style>""", unsafe_allow_html=True)


with st.sidebar:
    option = option_menu("Menu", ["Home", "Setup", "Load Data", "Analyze", "Train Model", "Model Monitoring", "Model Catalog",
                                                            "Inference", "Inference Runs", "Clean up"],
                            icons=['house-fill', 'tags', 'download','graph-up', 'play-circle', '', 'list-task', 'boxes', 'speedometer2', ''],
                            menu_icon="menu-button-wide", default_index=0,
                            styles={
            "container": {"padding": "5!important", "background-color": "white","font-color": "#249dda"},
            "icon": {"color": "#31c0e7", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "white"},
            "nav-link-selected": {"background-color": "7734f9"},
        })


from Home import *

if option == 'Home':    
    main()

elif option == "Setup":
    session = create_snowflake_connexion()
    st.subheader('Setup')

    with st.expander("Setup Database and needed objects"):
        if st.button(' ▶️  Setup database'):
            f = open("../sql/init_database.sql", "r")
            init_sql = f.read()
            tab = init_sql.split(";")
            for statement in tab:
                if statement and statement != "":
                    session.sql(statement).collect()
                    st.write(statement)            

    st.markdown('----')

elif option == "Load Data":
    session = create_snowflake_connexion()

    st.subheader('Load Data')
    #data = session.sql("SELECT * FROM TRAIN_DATASET").to_pandas()
    #st.write(data)

    import pandas as pd
    import zipfile

    with st.expander("Step 1 - Load Train Dataset"):
        if st.button(' ▶️  Start loading train data'):
            # open zipped dataset
            with zipfile.ZipFile("../data/TRAIN_DATASET.zip") as z:
            # open the csv file in the dataset
                with z.open("TRAIN_DATASET.csv") as f:        
                    # read the dataset
                    pandas_df = pd.read_csv(f)
                    #pandas_df = pd.read_csv("../data/Dataset/TRAIN_DATASET.csv")
                    #snowpark_df = session.create_dataframe(pandas_df)
                    session.write_pandas(pandas_df, "TRAIN_DATASET", auto_create_table=False, overwrite=True)
                    #st.write(pandas_df)
                    st.write("CSV File TRAIN_DATASET.csv loaded correctly into Snowflake Table TRAIN_DATASET")

    with st.expander("Step 2 - Load Test Dataset"):
        if st.button(' ▶️  Start loading test data'):
            with zipfile.ZipFile("../data/TEST_DATASET.zip") as z:
                # open the csv file in the dataset
                with z.open("TEST_DATASET.csv") as f:        
                    # read the dataset
                    pandas_df = pd.read_csv(f)
                    #pandas_df = pd.read_csv("../data/Dataset/TRAIN_DATASET.csv")
                    #snowpark_df = session.create_dataframe(pandas_df)
                    session.write_pandas(pandas_df, "TEST_DATASET", auto_create_table=False, overwrite=True)
                    #st.write(pandas_df)
                    st.write("CSV File TEST_DATASET.csv loaded correctly into Snowflake Table TEST_DATASET")

    st.markdown('----')

elif option == "Analyze":
    session = create_snowflake_connexion()

    with st.container():
        df_tables = session.table('information_schema.tables').filter(col("table_schema") == schema_name).select(col("table_name"), col("row_count"), col("created"))
        pd_tables = df_tables.to_pandas()
        
        st.subheader('Tables available')
        st.dataframe(pd_tables)
        
    with st.container():
        
        list_tables_names = pd_tables["TABLE_NAME"].values.tolist()
        st.subheader('Analyze Dataset')
        table_to_print = st.selectbox("Select table to describe statistics :", list_tables_names)
        
        if (table_to_print):
            table_to_print = schema_name + "." + table_to_print
        
            df_table = session.table(table_to_print)

            pd_table = df_table.limit(10).to_pandas()
            pd_describe = df_table.describe().to_pandas()
            
            with st.expander("Statistics", False):
                col0, col1, col2 = st.columns(3)

                with col0:
                    total = df_table.count()
                    st.metric(label="Total", value=total)

                with col1:
                    positive = df_table.filter(col(target_column_name) == 'positive').count()
                    st.metric(label="Positive", value=positive)

                with col2:                
                    negative = df_table.filter(col(target_column_name) == 'negative').count()
                    st.metric(label="Negative", value=negative)

            with st.expander("Sample Data", False):
                st.subheader(table_to_print)
                st.dataframe(pd_table)

            with st.expander("Data Description", False):
                st.subheader('Data Description')
                st.dataframe(pd_describe)
        
    st.markdown('----')

elif option == "Train Model":
    session = create_snowflake_connexion()

    with st.container():
        
        st.subheader("Train Dataset")

        df_tables = session.table('information_schema.tables').filter(col("table_schema") == schema_name).select(col("table_name"))
        pd_tables = df_tables.to_pandas()
        
        list_tables_names = pd_tables["TABLE_NAME"].values.tolist()
        table_to_train = st.selectbox("Select table to train model :", list_tables_names)
        
        if (table_to_train):
            table_to_train = schema_name + "." + table_to_train

            #with st.container():
                #st.write("Table selected : " + table_to_train)

            with st.container():
                st.subheader("Configuration")

                cwh_size_option = st.selectbox(
                'To change the WH size, select one :',
                ('Medium', 'Large', 'X-Large', '2X-Large', '3X-Large', '4X-Large'))
                st.write('You selected:', cwh_size_option)
                
                cwh = session.sql("select current_warehouse()").collect()
                cwh = str(cwh[0])
                cwh = cwh.replace("CURRENT_WAREHOUSE","").replace(")", "").replace("Row((=","")\
                            .replace("'","")
                
                cmd = "show warehouses like '" + cwh + "'"
                cwh_size = session.sql(cmd).collect()
                
                if cwh_size_option:
                    cmd = "ALTER WAREHOUSE " + cwh + " set warehouse_size = '" + cwh_size_option + "'"
                    session.sql(cmd).collect()
                    cwh_size = cwh_size_option
                else:
                    cwh_size = cwh_size[0]["size"]
                
                #with st.container():
                with st.expander("See Current Configuration", False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write ('Table selected :')
                        st.write ('Algorithm used :')
                        st.write('Virtual Warehouse :')
                        st.write('Size VH :')

                    with col2:
                        st.write(table_to_train)
                        st.write ('SVM')
                        st.write(cwh)
                        st.write(cwh_size)
                        
                st.subheader("Run Model")
                #col1, col2 = st.columns(2)
    
                with st.container():
                    col1, col2 = st.columns([1, 0.2])
                    with col1:
                        use_optimized = st.checkbox('Use Optimized Warehouse for Large Trainings')
                    
                    with col2:
                        st.button('▶️  Train Model', on_click=train_model, args=(session, table_to_train, 
                                    cwh, cwh_size, use_optimized))
                        
    st.markdown('----')
        
elif option == 'Model Monitoring':
    session = create_snowflake_connexion()

    st.subheader('Monitoring')
    df_query_history = session.sql("SELECT * FROM table(information_schema.query_history());").to_pandas()

    st.dataframe(df_query_history, use_container_width=True)

    c1, c2 = st.columns([1, 0.15])
    with c2:
            ref_btn = st.button('▶️  Refresh')
            if ref_btn:
                st.experimental_rerun()

    st.markdown('----')

elif option == "Model Catalog":
    session = create_snowflake_connexion()

    st.subheader('Models')
    
    with st.container():
        data = session.sql("LIST @MODELS")
        st.write(data)

    st.markdown('----')

elif option == "Inference":
    session = create_snowflake_connexion()

    subtab_test_dataset, subtab_accuracy = st.tabs(['Test Dataset', 'Accuracy'])

    with subtab_test_dataset:
        st.subheader('Statistics')
        test_dataset = session.table("TEST_DATASET")
        new_df = test_dataset.withColumn("SENTIMENT_FLAG", fn.when(test_dataset.SENTIMENT == "positive", 1)
                                            .otherwise(2))        
        df_predict = new_df.select(new_df.REVIEW, new_df.SENTIMENT, new_df.SENTIMENT_FLAG, \
                fn.call_udf("predict_review", fn.array_construct(col("REVIEW"), col("SENTIMENT_FLAG"))).alias('PREDICTED_REVIEW'))
        df_predict.write.mode('overwrite').saveAsTable('REVIEW_PREDICTION')
        #st.write(new_df.count())
        
        col0, col1, col2 = st.columns(3)
        
        with st.container():
            with col0 :
                total = new_df.count()
                st.metric(label="Total", value=total)
        
            with col1:
                positive = df_predict.filter(col(target_column_name) == 'positive').count()
                st.metric(label="Positive", value=positive)

            with col2:                
                negative = df_predict.filter(col(target_column_name) == 'negative').count()
                st.metric(label="Negative", value=negative)

        st.markdown('----')

        st.subheader('Sample Data')
        st.dataframe(new_df.to_pandas())

    with subtab_accuracy:
        st.subheader('Score')        
        df_predict = session.table("REVIEW_PREDICTION")

        #df_score = assess_performance_df(df_predict.toPandas().PREDICTED_REVIEW, df_predict.toPandas().SENTIMENT_FLAG)
        #st.dataframe(df_score)
        Accuracy, Recall, Precision, F1_Score = assess_performance(df_predict.toPandas().PREDICTED_REVIEW, df_predict.toPandas().SENTIMENT_FLAG)

        col0, col1, col2, col3 = st.columns(4)
        with st.container():
            with col0:
                st.metric(label="Accuracy", value=Accuracy)

            with col1:
                st.metric(label="Recall", value=Recall)

            with col2:
                st.metric(label="Precision", value=Precision)

            with col3:
                st.metric(label="F1_Score", value=F1_Score)

    st.markdown('----')

elif option == "Inference Runs":
    session = create_snowflake_connexion()

    with st.container():
        st.subheader('Prediction')

        df_predict = session.sql("SELECT REVIEW, PREDICTED_REVIEW FROM REVIEW_PREDICTION")
        df_predict = df_predict.withColumn("PREDICTED_REVIEW_LABEL", fn.when(df_predict.PREDICTED_REVIEW == 1, "positive") \
                                            .otherwise("negative"))
        
        st.dataframe(df_predict.to_pandas())

elif option == "Clean up":
    session = create_snowflake_connexion()
    st.subheader('Clean up')

    with st.expander("Step - Clean up"):
        if st.button(' ▶️  Clean database'):
            session.sql("DROP DATABASE IF EXISTS IMDB").collect()
            st.write("Database and all related objects are properly cleaned")
            session.sql("DROP WAREHOUSE IF EXISTS DEMO_WH").collect()
            st.write("Warehouse DEMO_WH cleaned properly")
