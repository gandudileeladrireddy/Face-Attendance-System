import streamlit as st
import pandas as pd
import sqlite3
from utils import Database, DB_FILE

st.set_page_config(page_title="Database", layout="wide")
st.title("Database Management")
db = Database()

t1, t2 = st.tabs(["Users", "Logs"])

with t1:
    users = db.get_all_users()
    if users:
        df = pd.DataFrame(users)[['id', 'name']]
        st.dataframe(df, use_container_width=True)
        del_id = st.selectbox("Delete ID", df['id'])
        if st.button("Delete"):
            if db.delete_user(del_id):
                if 'known_users' in st.session_state: del st.session_state['known_users']
                st.rerun()

with t2:
    conn = sqlite3.connect(DB_FILE)
    df_logs = pd.read_sql("SELECT logs.emp_id, users.name, logs.timestamp FROM logs JOIN users ON logs.emp_id = users.emp_id ORDER BY timestamp DESC", conn)
    st.dataframe(df_logs, use_container_width=True)
    st.download_button("Download CSV", df_logs.to_csv(index=False), "logs.csv")