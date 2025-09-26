import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_all_residuals_with_events(df, residuals_test, MinC, MaxC, events, axes):
    fig, axs = plt.subplots(4, 2, figsize=(18, 20), sharex=True)  # 8 subplots (4 rows × 2 cols)
    axs = axs.flatten()

    for i, axis in enumerate(axes):
        ax = axs[i]
        r = residuals_test[axis]
        t = df['time'].values

        ax.plot(t, r, color="blue", linewidth=1, label="Residual")
        ax.axhline(MinC[axis], linestyle="--", color="orange", label="MinC")
        ax.axhline(MaxC[axis], linestyle="--", color="red", label="MaxC")

        for ev in events:
            if ev.axis != axis:
                continue
            # Event spans
            ax.axvline(ev.start_time, color="green" if ev.level=="ALERT" else "red",
                       linestyle="--", alpha=0.7)
            ax.axvline(ev.end_time, color="green" if ev.level=="ALERT" else "red",
                       linestyle="--", alpha=0.7)

            # Peak point
            ax.scatter(ev.end_time, ev.peak_residual,
                       color="red" if ev.level=="ERROR" else "orange", marker="x", s=60)

            # Duration label
            mid_time = ev.start_time + (ev.end_time - ev.start_time)/2
            ax.text(mid_time, ev.peak_residual + 1,
                    f"{ev.level}\n{ev.duration_sec:.1f}s",
                    color="red" if ev.level=="ERROR" else "orange",
                    ha="center", va="bottom", fontsize=7,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        # Axis formatting
        ax.set_title(f"{axis}: Residuals with Alerts/Errors", fontsize=10)
        ax.set_ylabel("Residual")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

        # Format x-axis only on bottom row
        if i >= len(axes) - 2:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    fig.suptitle("Residuals with Alerts/Errors for All Axes", fontsize=14, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save single PNG
    out_path = "images/all_axes_residuals_with_alerts_errors.png"
    plt.savefig(out_path)
    plt.show()

    print(f"✅ Saved combined plot: {out_path}")
