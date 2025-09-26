import math, uuid
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class Event:
    event_id: str
    axis: str
    level: str       # ALERT / ERROR
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_sec: float
    mean_residual: float
    peak_residual: float

class ThresholdChecker:
    def __init__(self, MinC, MaxC, sample_interval_sec, T_seconds=10):
        self.MinC = MinC
        self.MaxC = MaxC
        self.sample_interval_sec = sample_interval_sec
        self.rows_required = max(1, math.ceil(T_seconds / sample_interval_sec))
        print(f"T = {T_seconds}s | {sample_interval_sec:.2f}s per row | rows_required={self.rows_required}")

    def detect_events(self, df, axes, models):
        events, residuals_test = [], {}
        X_test = df[['time_numeric']].values

        for axis in axes:
            y, yhat = df[axis].values, models[axis].predict(X_test)
            r = y - yhat
            residuals_test[axis] = r

            streak_type, streak_start = None, None

            for i, ri in enumerate(r):
                # Decide current level
                if ri >= self.MaxC[axis]:
                    current = "ERROR"
                elif ri >= self.MinC[axis]:
                    current = "ALERT"
                else:
                    current = None

                # Start streak
                if current and streak_type is None:
                    streak_type, streak_start = current, i

                # End streak
                elif not current and streak_type:
                    streak_len = i - streak_start
                    if streak_len >= self.rows_required:
                        seg = r[streak_start:i]
                        events.append(Event(
                            event_id=str(uuid.uuid4())[:8],
                            axis=axis,
                            level=streak_type,
                            start_time=df.loc[streak_start, 'time'],
                            end_time=df.loc[i-1, 'time'],
                            duration_sec=streak_len * self.sample_interval_sec,
                            mean_residual=float(np.nanmean(seg)),
                            peak_residual=float(np.nanmax(seg))
                        ))
                    streak_type, streak_start = None, None

            # Finalize streak if still open
            if streak_type:
                streak_len = len(r) - streak_start
                if streak_len >= self.rows_required:
                    seg = r[streak_start:]
                    events.append(Event(
                        event_id=str(uuid.uuid4())[:8],
                        axis=axis,
                        level=streak_type,
                        start_time=df.loc[streak_start, 'time'],
                        end_time=df.loc[len(r)-1, 'time'],
                        duration_sec=streak_len * self.sample_interval_sec,
                        mean_residual=float(np.nanmean(seg)),
                        peak_residual=float(np.nanmax(seg))
                    ))

        return events, residuals_test
