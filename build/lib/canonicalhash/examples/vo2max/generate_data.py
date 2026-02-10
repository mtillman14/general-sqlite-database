"""
Generate dummy VO2 max test data.

Simulates a standard ramp-protocol VO2 max test:
  - Duration: ~12 minutes (720 seconds)
  - Sampling: every 5 seconds (breath-by-breath averaged)
  - HR: resting ~75 bpm, rising to ~190 bpm
  - VO2: resting ~400 mL/min, rising to ~3800 mL/min with plateau at end

Outputs three CSV files in data/:
  - time_sec.csv
  - heart_rate_bpm.csv
  - vo2_ml_min.csv

Usage:
    python generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_vo2max_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate realistic VO2 max test data."""
    rng = np.random.default_rng(seed)

    # Time: 0 to 720 seconds, sampled every 5 seconds (145 points)
    time_sec = np.arange(0, 725, 5, dtype=float)
    n = len(time_sec)
    t_norm = time_sec / time_sec[-1]  # 0 to 1

    # Heart rate: sigmoid-like rise from ~75 to ~190 bpm
    # Starts slow, accelerates in the middle, approaches plateau at max
    hr_base = 75 + 115 * (1 / (1 + np.exp(-8 * (t_norm - 0.45))))
    hr_noise = rng.normal(0, 3, n)
    heart_rate = np.round(hr_base + hr_noise, 1)

    # VO2: linear rise with plateau in the last ~90 seconds
    # The plateau is the physiological signal that VO2max has been reached
    vo2_linear = 400 + 3400 * t_norm
    # Apply plateau via soft clamp near the end
    vo2_max_plateau = 3800.0
    vo2_base = np.where(
        vo2_linear < vo2_max_plateau,
        vo2_linear,
        vo2_max_plateau + 30 * np.log1p(vo2_linear - vo2_max_plateau),
    )
    # Breath-by-breath variability is substantial (~5-8% of current value)
    vo2_noise = rng.normal(0, 1, n) * (0.06 * vo2_base)
    vo2 = np.round(vo2_base + vo2_noise, 1)

    return time_sec, heart_rate, vo2


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    time_sec, heart_rate, vo2 = generate_vo2max_data()

    pd.DataFrame({"time_sec": time_sec}).to_csv(data_dir / "time_sec.csv", index=False)
    pd.DataFrame({"heart_rate_bpm": heart_rate}).to_csv(data_dir / "heart_rate_bpm.csv", index=False)
    pd.DataFrame({"vo2_ml_min": vo2}).to_csv(data_dir / "vo2_ml_min.csv", index=False)

    print(f"Generated {len(time_sec)} samples in {data_dir}/")
    print(f"  time:  {time_sec[0]:.0f} – {time_sec[-1]:.0f} sec")
    print(f"  HR:    {heart_rate.min():.0f} – {heart_rate.max():.0f} bpm")
    print(f"  VO2:   {vo2.min():.0f} – {vo2.max():.0f} mL/min")
