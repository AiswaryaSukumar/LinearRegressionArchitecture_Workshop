import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

class DataExtractor:
    def __init__(self):
        load_dotenv()
        self.engine = create_engine(
            f"postgresql+psycopg2://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}"
            f"@{os.getenv('PGHOST')}:{os.getenv('PGPORT','5432')}/{os.getenv('PGDATABASE')}"
            f"?sslmode={os.getenv('PGSSL','require')}"
        )

    def extract(self):
        query = '''
        SELECT time, axis1, axis2, axis3, axis4, axis5, axis6, axis7, axis8
        FROM staging_measurements
        '''
        df = pd.read_sql(query, self.engine)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        df['time_numeric'] = (df['time'] - df['time'].min()).dt.total_seconds()
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/Training_data.csv', index=False)
        return df
