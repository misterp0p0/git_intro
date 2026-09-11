#!/usr/bin/env python3
"""Self-tests for the MFI-20 excursion study. Run: python test_mfi_study.py"""

import sys

import numpy as np
import pandas as pd

from mfi_study import (
    money_flow_index,
    compare_mfi,
    format_mfi_check,
    load_csv,
    summarise,
    report,
    find_signals,
    measure_events,
    resample_bars,
    stop_survival,
)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def bars(rows, start="2024-01-02 18:00", freq="4h", tz="America/New_York"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz=tz)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)


# ---------------------------------------------------------------- MFI maths

def naive_mfi(df, length):
    """Deliberately unvectorised reference implementation."""
    tp = [(h + l + c) / 3 for h, l, c in zip(df["high"], df["low"], df["close"])]
    rmf = [t * v for t, v in zip(tp, df["volume"])]
    out = [float("nan")] * len(df)
    for i in range(length, len(df)):
        pos = neg = 0.0
        for k in range(i - length + 1, i + 1):
            if tp[k] > tp[k - 1]:
                pos += rmf[k]
            elif tp[k] < tp[k - 1]:
                neg += rmf[k]
        out[i] = float("nan") if pos + neg == 0 else 100.0 * pos / (pos + neg)
    return pd.Series(out, index=df.index)


def test_mfi_hand_computed():
    """A 3-bar MFI worked out by hand.

    TP: bar0 = 10, bar1 = 20, bar2 = 30, bar3 = 10
    vol = 1 throughout, so RMF == TP.
    Window of 3 ending at bar3 covers bars 1,2,3:
        bar1 up   -> pos += 20
        bar2 up   -> pos += 30
        bar3 down -> neg += 10
        MFI = 100 * 50 / 60 = 83.3333...
    """
    df = bars([
        [10, 10, 10, 10, 1],
        [20, 20, 20, 20, 1],
        [30, 30, 30, 30, 1],
        [10, 10, 10, 10, 1],
    ])
    mfi = money_flow_index(df, length=3)
    check("MFI hand-computed value",
          abs(mfi.iloc[3] - 100 * 50 / 60) < 1e-9, f"got {mfi.iloc[3]}")
    check("MFI warm-up is NaN", mfi.iloc[:3].isna().all())


def test_mfi_matches_naive():
    rng = np.random.default_rng(7)
    n = 400
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 2.0, n)
    low = close - rng.uniform(0.1, 2.0, n)
    df = bars(np.column_stack([close, high, low, close, rng.uniform(1e3, 1e5, n)]).tolist())
    for length in (9, 14, 21):
        a = money_flow_index(df, length)
        b = naive_mfi(df, length)
        both = a.notna() & b.notna()
        check(f"MFI matches reference impl (len={length})",
              bool(both.sum() > 300) and np.allclose(a[both], b[both]),
              f"n={int(both.sum())}")


def test_mfi_extremes():
    up = bars([[i, i + 1, i - 1, i, 100] for i in range(1, 30)])
    check("monotonic rally -> MFI 100", abs(money_flow_index(up, 14).iloc[-1] - 100) < 1e-9)
    down = bars([[i, i + 1, i - 1, i, 100] for i in range(30, 1, -1)])
    check("monotonic selloff -> MFI 0", abs(money_flow_index(down, 14).iloc[-1]) < 1e-9)
    check("MFI bounded 0..100",
          bool(money_flow_index(up, 14).dropna().between(0, 100).all()))


# ------------------------------------------------------------ signal arming

def test_signal_arming():
    s = pd.Series([50, 19, 18, 17, 50, 60, 15, 10, 55])
    check("one signal per oversold episode", find_signals(s, 20, 20) == [1, 6])

    s2 = pd.Series([50, 19, 21, 19, 50])
    check("re-arms after closing back above threshold",
          find_signals(s2, 20, 20) == [1, 3])

    s3 = pd.Series([50, 19, 21, 19, 50])
    check("higher re-arm level suppresses the whipsaw",
          find_signals(s3, 20, 30) == [1])

    s4 = pd.Series([np.nan, np.nan, 10, 50])
    check("NaN warm-up ignored", find_signals(s4, 20, 20) == [2])

    check("no signal when never oversold",
          find_signals(pd.Series([50, 40, 30, 25]), 20, 20) == [])


# ------------------------------------------------- excursion measurement

def hand_case():
    """Signal on bar 3 (close 100). Price drops to 90, then recovers above 100.

    bar 3: close 100            <- signal, reference = 100
    bar 4: low 95   high  99    (+1 bar)
    bar 5: low 90   high  96    <- lowest low = 90 => 10 points, +2 bars
    bar 6: low 92   high  98    (+3 bars)
    bar 7: low 97   high 101    <- first high >= 100: recovery, +4 bars
    """
    return bars([
        [100, 101, 99, 100, 10],
        [100, 101, 99, 100, 10],
        [100, 101, 99, 100, 10],
        [100, 101, 99, 100, 10],
        [ 99,  99, 95,  96, 10],
        [ 96,  96, 90,  92, 10],
        [ 92,  98, 92,  97, 10],
        [ 97, 101, 97, 100, 10],
        [100, 105, 99, 104, 10],
    ])


def test_excursion_basics():
    df = hand_case()
    mfi = pd.Series([np.nan] * 9, index=df.index)
    mfi.iloc[3] = 15.0            # force exactly one signal on bar 3
    mfi.iloc[4:] = 50.0

    ev = measure_events(df, mfi, "NQ", threshold=20, rearm=20)
    check("one event produced", len(ev) == 1, f"got {len(ev)}")
    r = ev.iloc[0]
    check("reference = signal bar close", r["reference_price"] == 100)
    check("lowest low found", r["lowest_low"] == 90)
    check("points dropped = 10", r["points_dropped"] == 10)
    check("bars to low = +2", r["bars_to_low"] == 2, f"got {r['bars_to_low']}")
    check("bars to recovery = +4", r["bars_to_recovery"] == 4, f"got {r['bars_to_recovery']}")
    check("hours to recovery = 16", abs(r["hours_to_recovery"] - 16.0) < 1e-9,
          f"got {r['hours_to_recovery']}")
    check("marked recovered", bool(r["recovered"]))
    check("NQ dollars = 10 pts * $20", r["dollars_dropped_full"] == 200)
    check("NQ micro dollars = 10 pts * $2", r["dollars_dropped_micro"] == 20)
    check("pct dropped = 10%", abs(r["pct_dropped"] - 10.0) < 1e-9)


def test_excursion_excludes_post_recovery_lows():
    """A deeper low *after* recovery must not be counted."""
    df = bars([
        [100, 101, 99, 100, 10],   # 0 signal
        [100, 101, 95, 100, 10],   # 1 low 95
        [100, 105, 99, 104, 10],   # 2 recovery (high 105 >= 100)
        [100, 101, 50, 60, 10],    # 3 crash, must be ignored
    ])
    mfi = pd.Series([15.0, 50.0, 50.0, 50.0], index=df.index)
    ev = measure_events(df, mfi, "ES", threshold=20, rearm=20)
    check("low after recovery ignored", ev.iloc[0]["points_dropped"] == 5,
          f"got {ev.iloc[0]['points_dropped']}")


def test_no_adverse_move_is_zero_not_negative():
    df = bars([
        [100, 101, 99, 100, 10],   # signal, ref 100
        [101, 104, 101, 103, 10],  # never trades below 100, high >= 100
    ])
    mfi = pd.Series([15.0, 50.0], index=df.index)
    ev = measure_events(df, mfi, "GC", threshold=20, rearm=20)
    check("no adverse move clamps to 0", ev.iloc[0]["points_dropped"] == 0.0)


def test_signal_bar_inclusion():
    df = bars([
        [100, 101, 80, 100, 10],   # signal bar itself dipped to 80
        [100, 101, 99, 100, 10],
    ])
    mfi = pd.Series([15.0, 50.0], index=df.index)
    excl = measure_events(df, mfi, "NQ", 20, 20, include_signal_bar=False)
    incl = measure_events(df, mfi, "NQ", 20, 20, include_signal_bar=True)
    check("signal bar low excluded by default", excl.iloc[0]["points_dropped"] == 1.0)
    check("--include-signal-bar counts it", incl.iloc[0]["points_dropped"] == 20.0)


def test_unrecovered_is_flagged_not_dropped():
    df = bars([
        [100, 101, 99, 100, 10],
        [ 99,  99, 90,  91, 10],
        [ 91,  92, 85,  86, 10],
    ])
    mfi = pd.Series([15.0, 50.0, 50.0], index=df.index)
    ev = measure_events(df, mfi, "NQ", 20, 20)
    r = ev.iloc[0]
    check("unrecovered event retained", len(ev) == 1)
    check("unrecovered flagged", not bool(r["recovered"]))
    check("unrecovered reason recorded", "unrecovered" in r["censored_reason"])
    check("unrecovered bars_to_recovery = -1", r["bars_to_recovery"] == -1)
    check("open drawdown still measured", r["points_dropped"] == 15.0)


def test_max_lookahead_censors():
    df = bars([[100, 101, 99, 100, 10]]
              + [[99, 99, 95, 96, 10]] * 10
              + [[96, 120, 96, 119, 10]])
    mfi = pd.Series([15.0] + [50.0] * 11, index=df.index)
    capped = measure_events(df, mfi, "NQ", 20, 20, max_lookahead=5)
    check("lookahead cap censors late recovery", not bool(capped.iloc[0]["recovered"]))
    uncapped = measure_events(df, mfi, "NQ", 20, 20)
    check("uncapped finds the recovery", bool(uncapped.iloc[0]["recovered"]))


def test_next_open_reference():
    df = bars([
        [100, 101, 99, 100, 10],   # signal, close 100
        [ 98, 101, 90,  95, 10],   # next open 98 is the reference
    ])
    mfi = pd.Series([15.0, 50.0], index=df.index)
    ev = measure_events(df, mfi, "NQ", 20, 20, reference="next_open")
    r = ev.iloc[0]
    check("next_open reference used", r["reference_price"] == 98)
    check("next_open drop = 98 - 90", r["points_dropped"] == 8)


def test_close_vs_touch_recovery():
    """Upper wick clips the reference but the bar closes below it.

    bar 0: signal, close 100
    bar 1: high 101 (touches back above 100) but closes 97, low 96
    bar 2: low 80, high 90
    bar 3: closes 104  <- first CLOSE back above 100
    touch rule  -> recovery at bar 1, drop = 100 - 96 = 4
    close rule  -> recovery at bar 3, drop = 100 - 80 = 20
    """
    df = bars([
        [100, 101,  99, 100, 10],
        [100, 101,  96,  97, 10],
        [ 97,  90,  80,  85, 10],
        [ 85, 105,  85, 104, 10],
    ])
    mfi = pd.Series([15.0, 50.0, 50.0, 50.0], index=df.index)

    t = measure_events(df, mfi, "NQ", 20, 20, recovery="touch").iloc[0]
    c = measure_events(df, mfi, "NQ", 20, 20, recovery="close").iloc[0]

    check("touch rule recovers on bar 1", t["bars_to_recovery"] == 1)
    check("touch rule drop = 4", t["points_dropped"] == 4, f"got {t['points_dropped']}")
    check("close rule recovers on bar 3", c["bars_to_recovery"] == 3)
    check("close rule drop = 20", c["points_dropped"] == 20, f"got {c['points_dropped']}")

    check("both rules recorded on every row regardless of primary",
          t["points_dropped_touch"] == 4 and t["points_dropped_close"] == 20
          and c["points_dropped_touch"] == 4 and c["points_dropped_close"] == 20)
    check("close rule is never shallower than touch rule",
          c["points_dropped"] >= t["points_dropped"])


def test_close_rule_never_understates():
    """Across random data the close rule must dominate the touch rule."""
    rng = np.random.default_rng(3)
    n = 1500
    close = 5000 + np.cumsum(rng.normal(0, 6, n))
    high = close + rng.uniform(1, 12, n)
    low = close - rng.uniform(1, 12, n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    df = bars(np.column_stack([open_, high, low, close, rng.uniform(1e3, 9e4, n)]).tolist())
    mfi = money_flow_index(df, 14)
    # A symmetric random walk rarely drives MFI under 20, so this property
    # test uses a looser threshold purely to gather a usable sample.
    ev = measure_events(df, mfi, "ES", 35, 35)
    check("random-data signals found", len(ev) > 10, f"got {len(ev)}")
    both = ev[ev["recovered_touch"] & ev["recovered_close"]]
    check("close-rule drop >= touch-rule drop on every event",
          bool((both["points_dropped_close"] >= both["points_dropped_touch"] - 1e-9).all()))
    check("close-rule recovery is never earlier",
          bool((both["bars_to_recovery_close"] >= both["bars_to_recovery_touch"]).all()))


# ---------------------------------------------------------------- plumbing

def test_resample_anchoring():
    idx = pd.date_range("2024-01-02 18:00", periods=24, freq="1h", tz="America/New_York")
    hourly = pd.DataFrame({
        "open": np.arange(24.0), "high": np.arange(24.0) + 1,
        "low": np.arange(24.0) - 1, "close": np.arange(24.0),
        "volume": np.ones(24) * 10,
    }, index=idx)
    out = resample_bars(hourly, "4h", session_start_hour=18)
    check("24h of 1h bars -> 6 4h bars", len(out) == 6, f"got {len(out)}")
    check("first bar anchored at 18:00", out.index[0].hour == 18, f"got {out.index[0]}")
    check("OHLC aggregation correct",
          out.iloc[0]["open"] == 0 and out.iloc[0]["close"] == 3
          and out.iloc[0]["high"] == 4 and out.iloc[0]["low"] == -1)
    check("volume summed", out.iloc[0]["volume"] == 40)

    midnight = resample_bars(hourly, "4h", session_start_hour=0)
    check("anchor choice changes the bars",
          midnight.index[0].hour != out.index[0].hour)


def test_summary_exposes_every_reported_field():
    """Regression guard: the report indexes these keys by name."""
    df = hand_case()
    mfi = pd.Series([np.nan] * 9, index=df.index)
    mfi.iloc[3] = 15.0
    mfi.iloc[4:] = 50.0
    s = summarise(measure_events(df, mfi, "NQ", 20, 20), "NQ")
    need = [
        "symbol", "signals", "resolved", "unresolved", "pts_median", "pts_mean",
        "pts_p75", "pts_p90", "pts_p95", "pts_max", "point_value_usd",
        "usd_median_full", "usd_p90_full", "usd_median_micro", "usd_p90_micro",
        "pct_median", "pct_p90", "bars_to_low_median", "bars_to_recovery_median",
        "bars_to_recovery_p90", "no_adverse_move_pct", "touch_pts_median",
        "touch_pts_p90", "touch_bars_median", "touch_resolved_next_bar_pct",
        "close_pts_median", "close_pts_p90", "close_pts_max", "close_bars_median",
    ]
    missing = [k for k in need if k not in s]
    check("summary exposes every reported field", not missing, f"missing {missing}")


def test_report_renders():
    """The whole report must render without a KeyError on any symbol."""
    import argparse
    df = hand_case()
    mfi = pd.Series([np.nan] * 9, index=df.index)
    mfi.iloc[3] = 15.0
    mfi.iloc[4:] = 50.0
    events = {s: measure_events(df, mfi, s, 20, 20) for s in ("NQ", "ES", "GC")}
    args = argparse.Namespace(timeframe="4h", mfi_length=14, threshold=20.0,
                              rearm=20.0, reference="signal_close",
                              recovery="touch", tz="America/New_York",
                              stop_levels=None)
    try:
        text = report(events, args)
        ok, detail = ("HEADLINE" in text and "TOUCH-BACK" in text), ""
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    check("full report renders", ok, detail)


def test_mfi_cross_check():
    idx = pd.date_range("2024-01-02 18:00", periods=50, freq="4h", tz="America/New_York")
    a = pd.Series(np.linspace(5, 95, 50), index=idx)

    same = compare_mfi(a, a.copy())
    check("identical series -> zero diff", same["max_abs_diff"] == 0)
    check("identical series -> MATCH", "MATCH" in format_mfi_check("NQ", same))

    off = compare_mfi(a, a + 0.005)
    check("float noise still MATCH", "MATCH" in format_mfi_check("NQ", off))

    bad = compare_mfi(a, a + 20.0)
    check("large divergence -> MISMATCH", "MISMATCH" in format_mfi_check("NQ", bad))
    check("MISMATCH explains what to check",
          "length is 14" in format_mfi_check("NQ", bad))

    nan = compare_mfi(a, pd.Series([np.nan] * 50, index=idx))
    check("no overlap handled", nan["compared"] == 0)
    check("no overlap message", "no overlapping" in format_mfi_check("NQ", nan))


def test_tradingview_export_shape(tmpdir="."):
    """A TradingView-shaped CSV must parse: ISO+offset time, mixed-case headers."""
    import os
    path = os.path.join(tmpdir, "_tv_probe.csv")
    idx = pd.date_range("2024-01-02 18:00", periods=30, freq="4h", tz="America/New_York")
    pd.DataFrame({
        "time": [t.isoformat() for t in idx],
        "open": np.arange(30.0) + 100, "high": np.arange(30.0) + 102,
        "low": np.arange(30.0) + 99, "close": np.arange(30.0) + 101,
        "Volume": np.full(30, 1000.0),
        "MFI": np.linspace(10, 90, 30),
    }).to_csv(path, index=False)
    try:
        df = load_csv(path, "America/New_York")
        check("TradingView headers parsed", set(["open", "high", "low", "close", "volume"]) <= set(df.columns))
        check("exported MFI column carried through", "mfi_ref" in df.columns)
        check("ISO+offset timestamps localised", str(df.index.tz) == "America/New_York")
        check("row count preserved", len(df) == 30, f"got {len(df)}")
    finally:
        os.remove(path)


def test_stop_survival_monotonic():
    ev = pd.DataFrame({
        "recovered": [True] * 5,
        "points_dropped": [1.0, 5.0, 10.0, 20.0, 100.0],
    })
    t = stop_survival(ev, [0.5, 5.0, 10.0, 100.0])
    check("survival rises with stop distance",
          list(t["survived_pct"]) == sorted(t["survived_pct"]))
    check("survival at 5 pts = 40%", abs(t.iloc[1]["survived_pct"] - 40.0) < 1e-9)
    check("survival at 100 pts = 100%", abs(t.iloc[3]["survived_pct"] - 100.0) < 1e-9)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(f"\n{t.__name__}")
        t()
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
