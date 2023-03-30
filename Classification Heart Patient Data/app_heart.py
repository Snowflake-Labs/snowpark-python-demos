
# Snowpark
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import avg, sum, col,lit, as_double
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import json

# Create Session object
@st.cache_resource
def create_session_object():
    
    with open('creds.json') as f:
        connection_parameters = json.load(f) 
    
    session = Session.builder.configs(connection_parameters).create()
    print(session.sql('select current_warehouse(), current_database(), current_schema()').collect())
    return session


def train (session, table, model, cwh, cwh_size, use_optimized, use_zero_copy_cloning):

    if (use_optimized):
        cmd = "alter warehouse " + cwh + " suspend"
        session.sql(cmd).collect()
    
        cmd = "alter warehouse " + cwh + " set warehouse_size = '2X-Large'"
        session.sql(cmd).collect()
    
        cmd = "alter warehouse " + cwh + " set WAREHOUSE_TYPE = 'SNOWPARK-OPTIMIZED'"
        session.sql(cmd).collect()
    
    model_name = str.replace(model, ' ', '_')
    session.call('sf_train',model, table, '@models', model_name, use_zero_copy_cloning)

    if (use_optimized):
        cmd = "alter warehouse " + cwh + " suspend"
        session.sql(cmd).collect()
    
        cmd = "alter warehouse " + cwh + " set WAREHOUSE_TYPE = 'STANDARD'"
        session.sql(cmd).collect()

        cmd = "alter warehouse " + cwh + " set warehouse_size = '" + cwh_size + "'"
        session.sql(cmd).collect()


def score (session, table_orig, model_name, target_table, cwh, cwh_size, size_wh):

    cmd = "alter warehouse " + cwh + " set warehouse_size = '" + size_wh + "'"
    session.sql(cmd).collect()
    
    session.call('sf_score', table_orig, target_table, '@models', model_name )
 
    cmd = "alter warehouse " + cwh + " set warehouse_size = '" + cwh_size + "'"
    session.sql(cmd).collect()

    
def copy_into (session, list_files, table_name):

    session.call('copy_into', list_files, table_name)

    
def to_pct(value):
    
    val1= (float(value) * 100)
    val2 = f'{val1:.2f}'
    
    return val2 + " %"

#########################################
##### MAIN STREAMLIT APP STARTS HERE ####
#########################################


st.set_page_config(page_title="HPD Classification",page_icon="❄️")

# Add header and a subheader
st.header("Classification Heart Patient Data")

session = create_session_object()

with st.sidebar:
    option = option_menu("Snowpark Classification Demo", ["Load Data", "Analyze", "Train Model", "Model Catalog",
                                                            "Inference", "Inference Runs"],
                            icons=['upload','graph-up', 'play-circle','list-task', 'boxes', 'speedometer2'],
                            menu_icon="snow", default_index=0,
                            styles={
            "container": {"padding": "5!important", "background-color": "white","font-color": "#249dda"},
            "icon": {"color": "#31c0e7", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "white"},
            "nav-link-selected": {"background-color": "7734f9"},
        })

if option == "Load Data":
    
    data_load = session.sql('ls @load_data').collect()

    st.markdown('----')
    st.subheader("Data Loading")
    
    col_files, col_name_table = st.columns(2)

    with st.container():
        with col_files:    # data loading
            list_files = []
            files_available = session.sql("ls @load_data").collect()
            for f in files_available:
                list_files.append(f["name"])
                            
            files = st.selectbox('Load data to train your models:',
                    list_files)
  
            st.write('Files to load:', files)

        with col_name_table:
            table_name = "DATA." + st.text_input ("Table name to be created:", value="DEFAULT")
            st.write('Table to be created:', table_name)

            
        files = "@" + files

        st.button('Load Data', on_click=copy_into, args=(session, files, table_name))

    st.markdown('----')
  
elif option == "Analyze":

    st.markdown('----')
    with st.container():
        df_tables = session.table('information_schema.tables').filter(col("table_schema") == 'DATA').select(col("table_name"), col("row_count"), col("created"))
        pd_tables = df_tables.to_pandas()
        
        st.subheader('Tables available:')
        st.dataframe(pd_tables)
        
    with st.container():
        
        list_tables_names = pd_tables["TABLE_NAME"].values.tolist()
        table_to_print = st.selectbox("Select table to describe statistics:", list_tables_names)
        
        if (table_to_print):      
            table_to_print = "DATA." + table_to_print

            df_table = session.table(table_to_print)

            pd_table = df_table.limit(3).to_pandas()
            pd_describe = df_table.describe().to_pandas()
    
            col1, col2 = st.columns(2)
            with st.container():
                with col1:
                    positive = df_table.filter(col('target') == 1).count()
                    st.metric(label="Positive", value=positive)

                with col2:                
                    negative = df_table.filter(col('target') == 0).count()
                    st.metric(label="Negative", value=negative)
            
            with st.container():
                st.subheader(table_to_print)
                st.dataframe(pd_table)
        
            with st.container():
                st.subheader('Data Description')
                st.dataframe(pd_describe)

elif option == "Train Model":
    
    with st.container():
        df_tables = session.table('information_schema.tables').filter(col("table_schema") == 'DATA').select(col("table_name"))
        pd_tables = df_tables.to_pandas()
        
        list_tables_names = pd_tables["TABLE_NAME"].values.tolist()
        table_to_train = st.selectbox("Select table to train model:", list_tables_names)
        
        if (table_to_train):
            
            table_to_train = "DATA." + table_to_train

            with st.container():
                st.text("Table selected: " + table_to_train)

            with st.container():
                
                
                df_models = session.table('models').select(col("model_name"))
                pd_models = df_models.to_pandas()
                    
                model_option = st.selectbox('Choose method for training:', pd_models)
                if (model_option):
                    st.write ('Model selected: ', model_option)
                
                    cwh = session.sql("select current_warehouse()").collect()
                    cwh = str(cwh[0])
                    cwh = cwh.replace("CURRENT_WAREHOUSE","").replace(")", "").replace("Row((=","")\
                                .replace("'","")

                    cmd = "show warehouses like '" + cwh + "'"
                    cwh_size = session.sql(cmd).collect()
                    cwh_size = cwh_size[0]["size"]
                     
                    col1, col2, col3 = st.columns(3)
                    with st.container():
                        with col1:
                            use_zero_copy_cloning = st.checkbox('Keep a zero-copy clone of training data')
                        with col2:
                            use_optimized = st.checkbox('Use Optimized Warehouse for Large Trainings')
                        with col3:
                            st.button('Train Model', on_click=train, args=(session, table_to_train, 
                                        model_option, cwh, cwh_size, use_optimized, use_zero_copy_cloning))
 

        st.markdown('----')
           
elif option == "Model Catalog":
    
    with st.container():
  
        df_accuracy = session.table('accuracy_sum_v')
        pd_accuracy = df_accuracy.to_pandas()

        st.subheader('Models Catalog')
        st.dataframe(pd_accuracy)       
        
    with st.container():
        df_top = df_accuracy.select(col("MODEL_NAME"), as_double(col("ACCURACY")).alias("ACCURACY")).sort(col("ACCURACY"), ascending=False).limit(5)
        pd_top = df_top.to_pandas()
        
        pd_top.set_index("MODEL_NAME", inplace = True)
        st.bar_chart(pd_top)
        
    
    with st.container():
    
        list_models = pd_accuracy["MODEL_NAME"]
        
        model = st.selectbox('Choose model for details:', list_models)
        
        pd_model = session.table('class_report_sumary_v')\
                    .filter(col("MODEL_NAME") == model)\
                    .to_pandas()
                
        col1, col2 = st.columns(2)
        with st.container():
            with col1:
                st.text(pd_model["MODEL_NAME"].values[0])
            with col2:
                st.text(pd_model["DATA_TRAINING"].values[0])
        
        st.markdown('----')
        
        col1, col2 = st.columns(2)
        with st.container():
            with col1:
                st.metric(label="True Positive", value=pd_model["TP"])
            with col2:
                st.metric(label="False Positive", value=pd_model["FP"])

        with st.container():
            with col1:
                 st.metric(label="False Negative", value=pd_model["FN"])            
            with col2:
                 st.metric(label="True Negative", value=pd_model["TN"])
               
            
        st.markdown('----')

        col1, col2, col3 = st.columns(3)
        with st.container():
            with col1:
                st.metric(label="Negative F1 Score", value=to_pct(pd_model["NEG_F1_SCORE"].values[0]))
            with col2:
                st.metric(label="Negative Precision", value=to_pct(pd_model["NEG_PRECISION"].values[0]))
            with col3:
                st.metric(label="Negative Recall", value=to_pct(pd_model["NEG_RECALL"].values[0]))

        with st.container():
            with col1:
                st.metric(label="Positive F1 Score", value=to_pct(pd_model["POS_F1_SCORE"].values[0]))
            with col2:
                st.metric(label="Positive Precision", value=to_pct(pd_model["POS_PRECISION"].values[0]))
            with col3:
                st.metric(label="Positive Recall", value=to_pct(pd_model["POS_RECALL"].values[0]))
        
        with st.container():
            st.metric(label="Accuracy", value=to_pct(pd_model["ACCURACY"].values[0]))

elif option == "Inference":
    st.markdown('----')
    
    cwh = session.sql("select current_warehouse()").collect()
    cwh = str(cwh[0])
    cwh = cwh.replace("CURRENT_WAREHOUSE","").replace(")", "").replace("Row((=","")\
                .replace("'","")

    cmd = "show warehouses like '" + cwh + "'"
    cwh_size = session.sql(cmd).collect()
    cwh_size = cwh_size[0]["size"]
    
    col_select_model, col_select_table, col_target_table = st.columns(3)
    
    with st.container():
        with col_select_model:
            df_accuracy = session.table('accuracy_sum_v')
            pd_accuracy = df_accuracy.to_pandas()

            list_models = pd_accuracy["MODEL_NAME"].values.tolist()
            model_name = st.selectbox("Select Model for Inference:", list_models)

        if (model_name):
            with col_select_table:
                df_tables = session.table('information_schema.tables').filter(col("table_schema") == 'DATA').select(col("table_name"), col("row_count"), col("created"))
                pd_tables = df_tables.to_pandas()
                list_tables = pd_tables["TABLE_NAME"].values.tolist()

                table_orig = "DATA." + st.selectbox("Select Table for Inference:", list_tables)

            with col_target_table:
                if (model_name != "") & (table_orig != ""):
                    def_output_value = table_orig + "_" + model_name + "_INF"
                else:
                    def_output_value = "OUTPUT"
                target_table = st.text_input ("Name output table:", value=def_output_value)
            
            col1, col2 = st.columns(2)
            with st.container():
                with col1:
                    size_wh = 'X-Small'
                    size_wh = st.selectbox("Select WH size:", ['X-Small', 'Small', 'Medium',
                                                    'Large', 'X-Large', '2X-Large'])
                with col2: 
                    st.button('Inference', on_click=score, args=((session, table_orig,
                                    model_name, target_table, cwh, cwh_size, size_wh)))

elif option == "Inference Runs":

    with st.container():
        df_inference_runs = session.table('inference_runs')
        pd_inference_runs = df_inference_runs.to_pandas()
        
        st.dataframe(pd_inference_runs)
    st.markdown('----')

    with st.container():
        
        df_inference_list = df_inference_runs.select(col("TARGET_TABLE"))
        pd_inference_list = df_inference_list.to_pandas()
        
        table_inference = st.selectbox("Select Inference Table for Details:", pd_inference_list)
        
        if (table_inference):
            df_detail_inference = df_inference_runs.filter(col("TARGET_TABLE") == table_inference)
            pd_detail_inference = df_detail_inference.to_pandas()
        
            col1, col2 = st.columns(2)
            with st.container():
                with col1:
                    st.metric(label="True Positive", value=pd_detail_inference["TP"])
                with col2:
                    st.metric(label="False Positive", value=pd_detail_inference["FP"])

            with st.container():
                with col1:
                     st.metric(label="False Negative", value=pd_detail_inference["FN"])            
                with col2:
                     st.metric(label="True Negative", value=pd_detail_inference["TN"])

            st.markdown('----')

            col1, col2, col3, col4 = st.columns(4)
            with st.container():
                with col1:
                    st.metric(label="ACCURACY", value = to_pct(pd_detail_inference["ACCURACY"]) )
                with col2:
                    st.metric(label="PRECISION", value = to_pct(pd_detail_inference["PRECISION"]) )
                with col3:
                    st.metric(label="RECALL", value = to_pct(pd_detail_inference["RECALL"]) )
                with col4:
                    st.metric(label="F1_SCORE", value = to_pct(pd_detail_inference["F1_SCORE"]) )



        
#if __name__ == "__main__":
#    session = create_session_object()

#   load_data(session)
