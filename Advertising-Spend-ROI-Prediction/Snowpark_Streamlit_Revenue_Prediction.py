import json
import altair as alt
import pandas as pd
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import col
import streamlit as st

APP_ICON_URL = "https://i.imgur.com/dBDOHH3.png"

def create_session():
    if "snowpark_session" not in st.session_state:
        session = Session.builder.configs(json.load(open("connection.json"))).create()
        st.session_state['snowpark_session'] = session
    else:
        session = st.session_state['snowpark_session']
    return session

@st.experimental_memo(show_spinner=False)
def load_data():
    historical_data = session.table("BUDGET_ALLOCATIONS_AND_ROI").unpivot("Budget", "Channel", ["SearchEngine", "SocialMedia", "Video", "Email"]).filter(col("MONTH") != "July")
    df_last_six_months_allocations = historical_data.drop("ROI").to_pandas()
    df_last_six_months_roi = historical_data.drop(["CHANNEL", "BUDGET"]).distinct().to_pandas()
    df_last_months_allocations = historical_data.filter(col("MONTH") == "June").to_pandas()
    return historical_data.to_pandas(), df_last_six_months_allocations, df_last_six_months_roi, df_last_months_allocations

st.set_page_config("SportsCo Ad Spend Optimizer", APP_ICON_URL, "centered")
st.write("<style>[data-testid='stMetricLabel'] {min-height: 0.5rem !important}</style>", unsafe_allow_html=True)
st.image(APP_ICON_URL, width=80)
st.title("SportsCo Ad Spend Optimizer")
session = create_session()
historical_data, df_last_six_months_allocations, df_last_six_months_roi, df_last_months_allocations = load_data()

st.header("Advertising budgets")
col1, _, col2 = st.columns([4, 1, 4])
channels = ["Search engine", "Social media", "Email", "Video"]
budgets = []
for channel, default, col in zip(channels, df_last_months_allocations["BUDGET"].values, [col1, col1, col2, col2]):
    with col:
        budget = st.slider(channel, 0, 100, int(default), 5)
        budgets.append(budget)

st.header("Predicted revenue")

@st.experimental_memo(show_spinner=False)
def predict(budgets):
    df_predicted_roi = session.sql(f"SELECT predict_roi(array_construct({budgets[0]*1000},{budgets[1]*1000},{budgets[2]*1000},{budgets[3]*1000})) as PREDICTED_ROI").to_pandas()
    predicted_roi, last_month_roi = df_predicted_roi["PREDICTED_ROI"].values[0] / 100000, df_last_six_months_roi["ROI"].iloc[-1]
    change = round((predicted_roi - last_month_roi) / last_month_roi * 100, 1)
    return predicted_roi, change

predicted_roi, change = predict(budgets)
st.metric("", f"$ {predicted_roi:.2f} million", f"{change:.1f} % vs last month")
months = ["January", "February", "March", "April", "May", "June", "July"]
july = pd.DataFrame({"MONTH": ["July", "July", "July", "July"], "CHANNEL": ["SEARCHENGINE", "SOCIALMEDIA", "VIDEO", "EMAIL"], "BUDGET": budgets, "ROI": [predicted_roi] * 4})
chart_data = historical_data.append(july).reset_index(drop=True)
chart_data = chart_data.replace(["SEARCHENGINE", "EMAIL", "SOCIALMEDIA", "VIDEO"], ["Search engine", "Email", "Social media", "Video"])
base = alt.Chart(chart_data).encode(alt.X("MONTH", sort=months, title=None))
bars = base.mark_bar().encode(
    y=alt.Y("BUDGET", title="Budget", scale=alt.Scale(domain=[0, 400])),
    color=alt.Color("CHANNEL", legend=alt.Legend(orient="top", title=" ")),
    opacity=alt.condition(alt.datum.MONTH == "July", alt.value(1), alt.value(0.3)),
)
lines = base.mark_line(size=3).encode(
    y=alt.Y("ROI", title="Revenue", scale=alt.Scale(domain=[0, 25])),
    color=alt.value("#808495"),
    tooltip=["ROI"],
)
points = base.mark_point(strokeWidth=3).encode(
    y=alt.Y("ROI"),
    stroke=alt.value("#808495"),
    fill=alt.value("white"),
    size=alt.condition(alt.datum.MONTH == "July", alt.value(300), alt.value(70)),
)
chart = alt.layer(bars, lines + points).resolve_scale(y="independent")
chart = chart.configure_view(strokeWidth=0).configure_axisY(domain=False).configure_axis(labelColor="#808495", tickColor="#e6eaf1", gridColor="#e6eaf1", domainColor="#e6eaf1", titleFontWeight=600, titlePadding=10, labelPadding=5, labelFontSize=14).configure_range(category=["#FFE08E", "#03C0F2", "#FFAAAB", "#995EFF"])
st.altair_chart(chart, use_container_width=True)

submitted = st.button("❄️ Save to Snowflake")
if submitted:
    with st.spinner("Making snowflakes..."):
        df = pd.DataFrame({"MONTH": ["July"], "SEARCHENGINE": [budgets[0]], "SOCIALMEDIA": [budgets[1]], "VIDEO": [budgets[2]], "EMAIL": [budgets[3]], "ROI": [predicted_roi]})
        session.write_pandas(df, "BUDGET_ALLOCATIONS_AND_ROI")
        # session.sql("INSERT INTO BUDGET_ALLOCATIONS_AND_ROI VALUES ('July'" + "," + str(budgets[0]) + "," + str(budgets[1]) + "," + str(budgets[2]) + "," + str(budgets[3]) + "," + str(predicted_roi) + ")").collect()
        st.success("✅ Successfully wrote budgets & prediction to your Snowflake account!")
        st.snow()
