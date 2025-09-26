import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


class Visualizer:
    def __init__(self, output_dir: str = "images"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def plot_regression_lines(self, df_train, models, axes, X):
        """Plot regression fits for all axes"""
        fig, axs = plt.subplots(4, 2, figsize=(15, 20), sharex=True)
        axs = axs.flatten()

        for i, axis in enumerate(axes):
            axs[i].scatter(df_train['time_numeric'], df_train[axis], s=5, alpha=0.5, label='Data')
            y_pred = models[axis].predict(X)
            axs[i].plot(df_train['time_numeric'], y_pred, color='red', label='Regression Line')
            axs[i].set_title(f'{axis}: Data vs Time')
            axs[i].set_ylabel(axis)
            axs[i].legend()
            axs[i].grid(alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "regression_fit_plots.png")
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Regression fit plots saved to {save_path}")

    def plot_residual_histograms(self, residuals_dict, axes):
        """Plot residual histograms for all axes"""
        fig, axs = plt.subplots(4, 2, figsize=(15, 20))
        axs = axs.flatten()

        for i, axis in enumerate(axes):
            axs[i].hist(residuals_dict[axis], bins=50, color='blue', alpha=0.7)
            axs[i].set_title(f'{axis}: Residual Histogram')
            axs[i].set_xlabel('Residual')
            axs[i].grid(alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "residual_histograms.png")
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Residual histograms saved to {save_path}")
