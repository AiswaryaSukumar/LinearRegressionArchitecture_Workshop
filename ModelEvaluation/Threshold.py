import numpy as np
import pandas as pd
import os

class ThresholdFinder:
    def __init__(self, output_dir: str = "Data"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def discover_thresholds(self, residuals_dict, method="percentile",
                            minc_q=97.5, maxc_q=99.5,
                            k_min=2.5, k_max=3.5):
        """
        Discover MinC and MaxC thresholds for each axis.
        - method="percentile": use quantiles (e.g., 95th, 98th)
        - method="std": use multiples of standard deviation
        """
        rows = []
        minc_by_axis = {}
        maxc_by_axis = {}

        for axis, res in residuals_dict.items():
            res = np.asarray(res)
            pos = res[res > 0]   # focus on positive residuals only
            if len(pos) == 0:
                pos = res  # fallback if all values <= 0

            if method == "percentile":
                MinC = float(np.percentile(pos, minc_q))
                MaxC = float(np.percentile(pos, maxc_q))
            elif method == "std":
                mu = float(np.mean(pos))
                sigma = float(np.std(pos, ddof=0))
                MinC = mu + k_min * sigma
                MaxC = mu + k_max * sigma
            else:
                raise ValueError("method must be 'percentile' or 'std'")

            minc_by_axis[axis] = MinC
            maxc_by_axis[axis] = MaxC
            rows.append({"axis": axis, "MinC": MinC, "MaxC": MaxC})

        return pd.DataFrame(rows), minc_by_axis, maxc_by_axis

    def save_thresholds(self, df_thresholds: pd.DataFrame, filename: str):
        """Save threshold table to CSV"""
        save_path = os.path.join(self.output_dir, filename)
        df_thresholds.to_csv(save_path, index=False)
        print(f"✅ Thresholds saved to {save_path}")
        return save_path
