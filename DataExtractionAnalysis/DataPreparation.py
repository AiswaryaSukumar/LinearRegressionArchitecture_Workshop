import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class DataPreparator:
    def __init__(self):
        # We can support both scaling methods
        self.std_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove missing values and ensure time_numeric exists."""
        df = df.dropna().copy()
        if "time_numeric" not in df.columns and "time" in df.columns:
            df["time_numeric"] = (df["time"] - df["time"].min()).dt.total_seconds()
        return df

    def normalize(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        """Apply StandardScaler + normalise for given features."""
        df_std = df.copy()
        df_norm = df.copy()

        df_std[features] = self.std_scaler.fit_transform(df_std[features])
        df_norm[features] = self.minmax_scaler.fit_transform(df_norm[features])

        return df_std, df_norm

    def split(self, df: pd.DataFrame, target: str, test_size=0.2, random_state=42):
        """Split data into train/test sets."""
        X = df.drop(columns=[target])
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: clean + normalization (returns standardized data by default)."""
        df = self.clean(df)
        features = [col for col in df.columns if col.startswith("axis")]
        df_std, _ = self.normalize(df, features)
        return df_std
