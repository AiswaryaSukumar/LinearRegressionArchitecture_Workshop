import pandas as pd
import numpy as np

class StreamingSimulator:
    def __init__(self, engine, models, axes, df_train, data_dir):
        self.engine = engine
        self.models = models
        self.axes = axes
        self.df_train = df_train
        self.data_dir = data_dir

    def generate_synthetic_data(self, make_synthetic_test, sample_interval_sec, MinC, MaxC):
        # Step 4A: Generate synthetic test data
        df_test, sample_interval_sec = make_synthetic_test(
            self.df_train, self.models, self.axes,
            n_rows=5000,
            sample_interval_sec=sample_interval_sec,
            residuals_dict={},
            force_above=True,
            MinC=MinC,
            MaxC=MaxC
        )
        print(f"Synthetic raw test data generated ({len(df_test)} rows).")

        # Step 4B: Normalize
        norm_df = df_test.copy()
        for axis in self.axes:
            mean_val = self.df_train[axis].mean()
            std_val = self.df_train[axis].std()
            if std_val > 0:
                norm_df[axis] = (df_test[axis] - mean_val) / std_val

        synthetic_path = self.data_dir / "Synthetic_test_normalized.csv"
        norm_df.to_csv(synthetic_path, index=False)
        print(f"✅ Normalized synthetic test data saved to {synthetic_path}")

        # Step 4C: Load into DB
        table_name = "synthetic_stream"
        norm_df.to_sql(table_name, self.engine, if_exists="replace", index=False)
        print(f"✅ Normalized synthetic test data loaded into DB (table: {table_name})")
        return table_name

    def stream_data(self, table_name, chunk_size=50):
        residuals_test = {axis: [] for axis in self.axes}
        query = f"SELECT * FROM {table_name} ORDER BY time"

        for batch_num, chunk in enumerate(pd.read_sql(query, self.engine, chunksize=chunk_size)):
            # Show only first 2 batches
            if batch_num < 2:
                print(f"Streaming batch {batch_num+1} (first 2 rows):")
                print(chunk.head(2))

            X_chunk = chunk[['time_numeric']].values
            for axis in self.axes:
                y_pred = self.models[axis].predict(X_chunk)
                y_actual = chunk[axis].values
                residuals_test[axis].extend(y_actual - y_pred)

        residuals_test = {axis: np.array(vals) for axis, vals in residuals_test.items()}
        print(f"✅ Residuals computed from streamed synthetic test data ({len(residuals_test[self.axes[0]])} rows total).")
        return residuals_test
