from dotenv import load_dotenv
import os
import psycopg2
import pandas as pd

load_dotenv()

def get_connection():
    conn_string = (
        f"host={os.getenv('DB_HOST')} "
        f"port={os.getenv('DB_PORT')} "
        f"dbname={os.getenv('DB_NAME')} "
        f"user={os.getenv('DB_USER')} "
        f"password={os.getenv('DB_PASS')} "
        f"sslmode=require"
    )
    return psycopg2.connect(conn_string)


def load_view(view_name):
    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM {view_name}", conn)
    conn.close()
    return df
