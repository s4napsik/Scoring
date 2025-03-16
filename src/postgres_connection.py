import pandas as pd
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
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

# Пример: Чтение данных из таблицы в Pandas DataFrame
try:
    df = pd.read_sql("SELECT * FROM prod.my_table;", engine)
    print(df.head())
except Exception as e:
    print(f"Error reading data: {e}")