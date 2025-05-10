import os
import urllib.parse

from sqlalchemy import create_engine, Table, insert, MetaData
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd


class SqlDb:
    def __init__(self, password=None, db_url=None, supabase_url=None, key=None):
        load_dotenv()

        self.password = password if password else urllib.parse.quote_plus(os.environ.get("DB_PASSWORD"))
        self.url = db_url if db_url else os.environ.get("DATABASE_URL").format(self.password)

        self.supabase_url = supabase_url if supabase_url else os.environ.get("SUPABASE_URL")
        self.key = key if key else os.environ.get("SUPABASE_KEY")

        self.engine = self._create_engine()  # sql alchemy engine
        self.db = self._create_db()  # llm wrapper for db usage
        self.client = create_client(self.supabase_url, self.key)  # supabase client - use for vectorstore

    def _create_engine(self):
        engine = create_engine(self.url)
        return engine

    def _create_db(self):
        db = SQLDatabase(self.engine)
        return db

    def upload_table_from_pandas_df(self, table_name, df, if_exists="append"):
        assert self.engine, "SQL Engine is not configured"
        assert table_name, "Please provide a table name"

        df.columns = df.columns.str.lower()
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)

    def upload_data_incrementally(self, table_name, data):
        metadata = MetaData()
        sql_table = Table(table_name, metadata, autoload_with=self.engine)

        with self.engine.connect() as connection:
            with connection.begin():
                stmt = insert(sql_table).values(data)
                connection.execute(stmt)

    def load_data_from_db(self, table_name):
        df = pd.read_sql_table(table_name, self.engine)
        return df
