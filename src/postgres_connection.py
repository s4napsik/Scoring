import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 10)
# Загрузка переменных окружения из файла .env
load_dotenv()
# Получение параметров из .env
HOST = os.getenv("HOST")
DB_PORT = os.getenv("PG_PORT")
PG_DB = os.getenv("PG_DB")
PG_LOGIN = os.getenv("PG_LOGIN")
PG_PASS = os.getenv("PG_PASS")

# Формирование строки подключения через SQLAlchemy
DATABASE_URL = f"postgresql+psycopg2://{PG_LOGIN}:{PG_PASS}@{HOST}:{DB_PORT}/{PG_DB}"

# Создание SQLAlchemy Engine
engine = create_engine(DATABASE_URL)

query = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'dwh' AND table_type = 'BASE TABLE'
"""

#Пример: Чтение данных из таблицы в Pandas DataFrame
def connection(table: str) -> pd.DataFrame:
    df = pd.read_sql(f"SELECT * FROM {table}", engine)
    return df

def all_tables() -> pd.DataFrame:
    df_tables = pd.read_sql(query, engine)
    return df_tables

