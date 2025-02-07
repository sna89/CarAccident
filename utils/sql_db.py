import os
import urllib.parse

from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv


class SqlDb:
    def __init__(self, password=None, url=None):
        load_dotenv()

        self.password = password if password else urllib.parse.quote_plus(os.environ.get("DB_PASSWORD"))
        self.url = url if url else os.environ.get("DATABASE_URL").format(self.password)

        self.engine = self._create_engine()
        self.db = self._create_db()

    def _create_engine(self):
        engine = create_engine(self.url)
        return engine

    def _create_db(self):
        db = SQLDatabase(self.engine)
        return db

    def upload_table_from_pandas_df(self, df, table_name):
        assert self.engine, "SQL Engine is not configured"
        assert table_name, "Please provide a table name"

        df.columns = df.columns.str.lower()
        df.to_sql(table_name, self.engine, if_exists='replace', index=False)


