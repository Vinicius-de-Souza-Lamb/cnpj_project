import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

@st.cache_resource
def _engine():
    secrets = st.secrets 

    user     = secrets.get("USER_POSTG"    , os.getenv("USER_POSTG"))
    pwd      = secrets.get("PASSWORD_POSTG", os.getenv("PASSWORD_POSTG"))
    host     = secrets.get("HOST_POSTG"    , os.getenv("HOST_POSTG", "localhost"))
    port     = secrets.get("PORT_POSTG"    , os.getenv("PORT_POSTG", "5432"))
    db       = secrets.get("DATABASE_POSTG", os.getenv("DATABASE_POSTG"))

    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True)

@st.cache_data(ttl=3600)
def load_df(sql: str) -> pd.DataFrame:
    return pd.read_sql(sql, _engine())