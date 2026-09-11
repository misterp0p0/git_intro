#!/usr/bin/env python3
"""Generate synthetic 1h futures bars so the pipeline can be exercised offline.

These are random walks with volatility clustering. They are ONLY for checking
that the tooling runs end to end -- the statistics they produce say nothing
about real NQ/ES/GC behaviour. Replace with real exported bars for real answers.
"""

import numpy as np
import pandas as pd


def synth(symbol, start_price, annual_vol, seed, hours=11600):
    rng = np.random.default_rng(seed)

    # Volatility clustering: an AR(1) process on log-variance.
    z = rng.standard_normal(hours)
    logvar = np.zeros(hours)
    for i in range(1, hours):
        logvar[i] = 0.97 * logvar[i - 1] + 0.25 * rng.standard_normal()
    per_hour = annual_vol / np.sqrt(252 * 23)
    sigma = per_hour * np.exp(0.5 * logvar)

    # Student-t shocks for fat tails.
    shocks = sigma * (z * 0.8 + 0.2 * rng.standard_t(4, hours))
    close = start_price * np.exp(np.cumsum(shocks))

    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(shocks) * close * rng.uniform(0.3, 1.4, hours)
    high = np.maximum(open_, close) + wick * rng.uniform(0, 1, hours)
    low = np.minimum(open_, close) - wick * rng.uniform(0, 1, hours)

    # Volume rises with range and follows a time-of-day profile.
    idx = pd.date_range("2024-01-02 18:00", periods=hours, freq="1h",
                        tz="America/New_York")
    tod = 1.0 + 1.8 * np.exp(-0.5 * ((idx.hour - 10) / 3.0) ** 2)
    rel_range = (high - low) / close
    volume = (rel_range / rel_range.mean()) * tod * rng.lognormal(9.5, 0.4, hours)

    df = pd.DataFrame({
        "datetime": idx, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume.round(),
    })
    path = f"demo_{symbol.lower()}_1h.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path}  {len(df)} bars  "
          f"{df['close'].min():,.1f}-{df['close'].max():,.1f}")
    return path


if __name__ == "__main__":
    synth("NQ", 21500, 0.22, seed=11)
    synth("ES", 5900, 0.15, seed=22)
    synth("GC", 2650, 0.16, seed=33)
