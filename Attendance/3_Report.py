import streamlit as st
import pandas as pd
from Home import face_rec


# Function to load logs from Redis database
def load_logs():
    # Retrieve the data from Redis Database
    redis_logs = face_rec.retrieve_logs()

    if redis_logs:
        # Convert the logs dictionary into a pandas DataFrame
        df_logs = pd.DataFrame.from_dict(redis_logs)
        st.dataframe(df_logs)
        st.success("Logs successfully retrieved from Redis")
    else:
        st.warning("No logs found in Redis")


st.set_page_config(page_title='Report')
st.subheader('Attendance Report')

# Load logs from Redis database
load_logs()
