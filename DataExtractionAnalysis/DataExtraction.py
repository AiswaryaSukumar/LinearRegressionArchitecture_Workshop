import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

class DataExtractor:
    def __init__(self):
        """Initialize extractor with DB credentials from .env"""
        load_dotenv()
        self.PGHOST = os.getenv('PGHOST')
        self.PGPORT = os.getenv('PGPORT', '5432')
        self.PGDATABASE = os.getenv('PGDATABASE')
        self.PGUSER = os.getenv('PGUSER')
        self.PGPASSWORD = os.getenv('PGPASSWORD')
        self.PGSSL = os.getenv('PGSSL', 'require')

        assert self.PGHOST and self.PGDATABASE and self.PGUSER and self.PGPASSWORD, \
            "❌ Missing DB env vars. Check your .env file!"

        self.engine = create_engine(
            f'postgresql+psycopg2://{self.PGUSER}:{self.PGPASSWORD}@{self.PGHOST}:{self.PGPORT}/{self.PGDATABASE}?sslmode={self.PGSSL}',
            pool_pre_ping=True
        )

    def load_data(self, query: str, save_path: str = "Data/Training_data.csv") -> pd.DataFrame:
        """Run SQL query, clean dataframe, and save to CSV"""
        df = pd.read_sql(query, self.engine)

        # Ensure datetime and sort
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values('time', inplace=True)

        # Add numeric time
        df['time_numeric'] = (df['time'] - df['time'].min()).dt.total_seconds()

        # Save to CSV
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

        print(f"✅ Data exported to {save_path}")
        return df


if __name__ == "__main__":
    extractor = DataExtractor()
    query = '''
    SELECT time, axis1, axis2, axis3, axis4, axis5, axis6, axis7, axis8
    FROM staging_measurements
    '''
    df_train = extractor.load_data(query)
    print(df_train.head())
