from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class ModelTrainer:
    def __init__(self, data_dir: str = "Data"):
        self.DATA_DIR = Path(data_dir)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.AXES = ['axis1', 'axis2', 'axis3', 'axis4', 'axis5', 'axis6', 'axis7', 'axis8']

        self.models = {}
        self.slopes = {}
        self.intercepts = {}
        self.residuals_dict = {}

    def load_training_data(self, filename: str = "Training_data.csv") -> pd.DataFrame:
        """Load the pre-extracted training data from CSV"""
        train_path = self.DATA_DIR / filename
        assert train_path.exists(), f"{filename} not found. Run DataExtraction first."

        df_train = pd.read_csv(train_path)
        df_train['time'] = pd.to_datetime(df_train['time'], errors='coerce')
        df_train = df_train.sort_values('time').reset_index(drop=True)

        # Ensure numeric time column
        if 'time_numeric' not in df_train.columns:
            df_train['time_numeric'] = (df_train['time'] - df_train['time'].min()).dt.total_seconds()

        # Drop rows with missing values
        df_train = df_train.dropna(subset=['time_numeric'] + self.AXES)

        # Estimate sampling interval
        time_diffs = df_train['time'].diff().dt.total_seconds()
        self.sample_interval_sec = float(np.nanmedian(time_diffs))
        print(f"✅ Estimated sampling interval: {self.sample_interval_sec:.3f} seconds")

        return df_train

    def fit_models(self, df_train: pd.DataFrame):
        """Fit linear regression models for each axis and calculate residuals"""
        X = df_train[['time_numeric']].values

        for axis in self.AXES:
            y = df_train[axis].values.astype(float)
            mdl = LinearRegression()
            mdl.fit(X, y)

            # Store results
            self.models[axis] = mdl
            self.slopes[axis] = float(mdl.coef_[0])
            self.intercepts[axis] = float(mdl.intercept_)

            # Compute residuals
            y_pred = mdl.predict(X)
            self.residuals_dict[axis] = y - y_pred

        print("✅ Models trained successfully.")

    def summarize_models(self) -> pd.DataFrame:
        """Return model slopes and intercepts"""
        model_summary = pd.DataFrame({
            'axis': self.AXES,
            'slope': [self.slopes[a] for a in self.AXES],
            'intercept': [self.intercepts[a] for a in self.AXES]
        }).sort_values('axis')

        model_summary.to_csv(self.DATA_DIR / "model_params.csv", index=False)
        print("📊 Model parameters saved to model_params.csv")
        return model_summary

    def compute_residual_stats(self) -> pd.DataFrame:
        """Compute residual mean and std per axis"""
        residual_stats = pd.DataFrame({
            'axis': self.AXES,
            'residual_mean': [float(np.mean(self.residuals_dict[a])) for a in self.AXES],
            'residual_std': [float(np.std(self.residuals_dict[a], ddof=0)) for a in self.AXES],
        }).sort_values('axis').reset_index(drop=True)

        residual_stats.to_csv(self.DATA_DIR / "residual_stats.csv", index=False)
        print("📊 Residual statistics saved to residual_stats.csv")
        return residual_stats


if __name__ == "__main__":
    trainer = ModelTrainer()
    df_train = trainer.load_training_data()
    trainer.fit_models(df_train)

    summary = trainer.summarize_models()
    stats = trainer.compute_residual_stats()

    print("\nModel Summary:\n", summary)
    print("\nResidual Stats:\n", stats)
