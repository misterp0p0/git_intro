#!/usr/bin/env python3
"""
MFI-20 downside-excursion study.

Question this answers
---------------------
On a 4-hour chart, every time the Money Flow Index (MFI) reaches an oversold
threshold (default 20), how many points does price drop *before* it trades back
up through the price it was at when MFI first hit 20?

For each signal we record the Maximum Adverse Excursion (MAE): the distance
from the reference price down to the lowest low printed before the first bar
whose high reaches back up to that reference price.

    reference price ......... close of the bar on which MFI first hit <= 20
    recovery ................ first later bar with high >= reference price
    points dropped (MAE) .... reference price - lowest low seen up to recovery

Usage
-----
    # Local CSVs (TradingView / NinjaTrader / Barchart / Databento exports)
    python mfi_study.py --csv NQ=nq_1h.csv ES=es_1h.csv GC=gc_1h.csv --outdir out

    # Already-4h bars: skip resampling
    python mfi_study.py --csv NQ=nq_4h.csv --no-resample --outdir out

    # Yahoo Finance (needs network access to query*.finance.yahoo.com)
    python mfi_study.py --yahoo NQ=NQ=F ES=ES=F GC=GC=F --period 730d --outdir out
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Contract specifications
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Contract:
    name: str
    point_value: float   # USD per 1.00 point, full-size contract
    tick_size: float
    micro_point_value: float

    @property
    def tick_value(self) -> float:
        return self.point_value * self.tick_size


CONTRACTS = {
    "NQ": Contract("E-mini Nasdaq-100", 20.0, 0.25, 2.0),
    "ES": Contract("E-mini S&P 500", 50.0, 0.25, 5.0),
    "GC": Contract("COMEX Gold", 100.0, 0.10, 10.0),
    "YM": Contract("E-mini Dow", 5.0, 1.00, 0.5),
    "RTY": Contract("E-mini Russell 2000", 50.0, 0.10, 5.0),
    "CL": Contract("NYMEX Crude Oil", 1000.0, 0.01, 100.0),
    "SI": Contract("COMEX Silver", 5000.0, 0.005, 1000.0),
}


def contract_for(symbol: str) -> Contract | None:
    """Best-effort match of a user label ('NQ', 'nq_4h', 'NQ=F') to a contract."""
    up = symbol.upper()
    for key in sorted(CONTRACTS, key=len, reverse=True):
        if up.startswith(key):
            return CONTRACTS[key]
    return None


# --------------------------------------------------------------------------
# Money Flow Index
# --------------------------------------------------------------------------

def money_flow_index(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Standard MFI.

    typical price  TP  = (H + L + C) / 3
    raw money flow RMF = TP * volume
    RMF is 'positive' on bars where TP rose vs the prior bar, 'negative' where
    it fell; bars with an unchanged TP contribute to neither side (this is the
    conventional treatment used by TradingView, StockCharts and Wilder).

        MFI = 100 * pos_sum / (pos_sum + neg_sum)

    which is algebraically identical to the more commonly quoted
    100 - 100 / (1 + pos_sum/neg_sum) but is well defined when neg_sum is 0.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    rmf = tp * df["volume"]

    delta = tp.diff()
    pos = rmf.where(delta > 0, 0.0)
    neg = rmf.where(delta < 0, 0.0)

    pos_sum = pos.rolling(length, min_periods=length).sum()
    neg_sum = neg.rolling(length, min_periods=length).sum()

    total = pos_sum + neg_sum
    mfi = 100.0 * pos_sum / total
    # A window with no money flow at all (flat typical price throughout) is
    # undefined; leave it as NaN rather than inventing a reading.
    mfi[total == 0] = np.nan
    # The first `length` bars have no diff for bar 0, so shift the warm-up by 1.
    mfi.iloc[: length] = np.nan
    return mfi.rename("mfi")


# --------------------------------------------------------------------------
# Data loading / shaping
# --------------------------------------------------------------------------

_COL_ALIASES = {
    "open": {"open", "o", "openprice"},
    "high": {"high", "h", "highprice", "max"},
    "low": {"low", "l", "lowprice", "min"},
    "close": {"close", "c", "closeprice", "last", "settle", "adj close", "adjclose"},
    "volume": {"volume", "vol", "v", "totalvolume", "size"},
}

# TradingView names an exported indicator after the study and its inputs,
# e.g. "MFI" or "Money Flow Index" or "MFI 14". Match loosely.
def _find_reference_mfi(df: pd.DataFrame):
    for col in df.columns:
        low = str(col).strip().lower()
        if low.startswith("mfi") or low.startswith("money flow"):
            return col
    return None


_TIME_ALIASES = ["datetime", "date_time", "timestamp", "time", "date", "bar_time", "ts_event"]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map an arbitrary vendor CSV onto open/high/low/close/volume."""
    lowered = {str(c).strip().lower(): c for c in df.columns}
    out = {}
    for canon, aliases in _COL_ALIASES.items():
        hit = next((lowered[a] for a in aliases if a in lowered), None)
        if hit is None:
            if canon == "volume":
                continue
            raise SystemExit(
                f"CSV is missing a '{canon}' column. Found: {list(df.columns)}"
            )
        out[canon] = df[hit]
    ref = _find_reference_mfi(df)
    if ref is not None:
        out["mfi_ref"] = df[ref]

    res = pd.DataFrame(out, index=df.index)
    if "volume" not in res.columns:
        raise SystemExit(
            "CSV has no volume column. MFI is a volume-weighted indicator and "
            "cannot be computed without it. Export bars that include volume "
            "(pass --volume-proxy to substitute tick count / constant volume, "
            "but understand that changes what the indicator measures)."
        )
    return res


def _parse_time_index(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Find the timestamp column (or use the existing index) and localise it."""
    lowered = {str(c).strip().lower(): c for c in df.columns}
    tcol = next((lowered[a] for a in _TIME_ALIASES if a in lowered), None)

    if tcol is not None:
        raw = df[tcol]
        if pd.api.types.is_numeric_dtype(raw):
            # Epoch seconds / millis / micros / nanos (TradingView, Databento).
            mx = float(pd.to_numeric(raw, errors="coerce").abs().max())
            unit = ("s" if mx < 1e11 else "ms" if mx < 1e14
                    else "us" if mx < 1e17 else "ns")
            idx = pd.to_datetime(raw, unit=unit, utc=True)
        else:
            idx = pd.to_datetime(raw, utc=True, format="mixed", errors="coerce")
        df = df.drop(columns=[tcol])
    else:
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")

    if pd.isna(idx).all():
        raise SystemExit("Could not parse any timestamps from the input file.")

    df = df.set_axis(pd.DatetimeIndex(idx, name="datetime"))
    df = df[~df.index.isna()]
    return df.tz_convert(tz).sort_index()


def load_csv(path: str, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _parse_time_index(df, tz)
    df = _normalise_columns(df)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0.0)
    return df[~df.index.duplicated(keep="last")]


def load_yahoo(symbol: str, period: str, interval: str, tz: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(symbol, period=period, interval=interval,
                      auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise SystemExit(f"Yahoo returned no data for {symbol}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    idx = pd.DatetimeIndex(raw.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    raw = raw.set_axis(idx).tz_convert(tz).sort_index()
    return _normalise_columns(raw)


DATABENTO_PRESETS = {
    "NQ": ("GLBX.MDP3", "NQ.c.0"),
    "ES": ("GLBX.MDP3", "ES.c.0"),
    "GC": ("GLBX.MDP3", "GC.c.0"),
    "YM": ("GLBX.MDP3", "YM.c.0"),
    "RTY": ("GLBX.MDP3", "RTY.c.0"),
    "CL": ("GLBX.MDP3", "CL.c.0"),
}


def load_databento(spec: str, start: str, end: str | None, interval: str,
                   tz: str, key: str | None = None) -> pd.DataFrame:
    """Fetch bars from Databento Historical.

    `spec` is either a bare root ('NQ', 'ES', 'GC') which expands via
    DATABENTO_PRESETS to the CME Globex front-month continuous contract, or an
    explicit 'DATASET:SYMBOL:STYPE' triple for full control, e.g.
    'GLBX.MDP3:NQZ5:raw_symbol'.

    The API key is read from the DATABENTO_API_KEY environment variable. Do not
    pass it on the command line -- it lands in your shell history.
    """
    import databento as db

    key = key or os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit(
            "Set DATABENTO_API_KEY in the environment, e.g.\n"
            "  export DATABENTO_API_KEY='db-...'"
        )

    parts = spec.split(":")
    if len(parts) == 1:
        root = parts[0].upper()
        if root not in DATABENTO_PRESETS:
            raise SystemExit(
                f"No Databento preset for {root!r}. Known: "
                f"{sorted(DATABENTO_PRESETS)}. Or give DATASET:SYMBOL:STYPE."
            )
        dataset, symbol = DATABENTO_PRESETS[root]
        stype_in = "continuous"
    elif len(parts) == 3:
        dataset, symbol, stype_in = parts
    else:
        raise SystemExit(f"Bad --databento spec {spec!r}; use ROOT or DATASET:SYMBOL:STYPE")

    schema = {"1h": "ohlcv-1h", "1m": "ohlcv-1m", "1d": "ohlcv-1d"}.get(interval)
    if schema is None:
        raise SystemExit(f"Databento interval must be 1m, 1h or 1d (got {interval!r})")

    client = db.Historical(key)
    store = client.timeseries.get_range(
        dataset=dataset, symbols=[symbol], stype_in=stype_in,
        schema=schema, start=start, end=end,
    )
    df = store.to_df()
    if df is None or df.empty:
        raise SystemExit(f"Databento returned no data for {spec} ({start}..{end}).")

    df = df.rename(columns=str.lower)
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df = df.set_axis(idx).tz_convert(tz).sort_index()
    return _normalise_columns(df)


def resample_bars(df: pd.DataFrame, rule: str, session_start_hour: int) -> pd.DataFrame:
    """Aggregate to `rule` bars anchored on the futures session open.

    CME equity-index and COMEX metals futures open at 18:00 New York time, and
    that is where charting platforms anchor intraday bars. Anchoring matters:
    4h buckets anchored at midnight produce different bars, a different MFI
    series, and therefore different signals.
    """
    anchor = (df.index[0].normalize() - pd.Timedelta(days=1)) + pd.Timedelta(hours=session_start_hour)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule, origin=anchor, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


# --------------------------------------------------------------------------
# Signal extraction and the excursion measurement
# --------------------------------------------------------------------------

@dataclass
class Event:
    symbol: str
    signal_time: pd.Timestamp
    signal_index: int
    mfi_at_signal: float
    min_mfi_after: float
    reference_price: float

    # Primary measurement, using whichever recovery rule was selected.
    lowest_low: float
    lowest_low_time: pd.Timestamp
    points_dropped: float
    pct_dropped: float
    dollars_dropped_full: float
    dollars_dropped_micro: float
    bars_to_low: int
    bars_to_recovery: int
    hours_to_recovery: float
    recovery_time: object
    recovered: bool
    censored_reason: str

    # Both rules are always recorded so the two can be compared directly.
    points_dropped_touch: float
    bars_to_recovery_touch: int
    recovered_touch: bool
    points_dropped_close: float
    bars_to_recovery_close: int
    recovered_close: bool


def find_signals(mfi: pd.Series, threshold: float, rearm: float) -> list[int]:
    """Positional indices of bars where MFI first reaches `threshold`.

    A signal fires on the bar where MFI closes at or below `threshold` while
    'armed'. The setup then disarms and only re-arms once MFI closes back above
    `rearm`, so a long stretch of MFI pinned under 20 counts as one event, not
    twenty. This is what "the MFI *first* hit 20" means.
    """
    vals = mfi.to_numpy(dtype=float)
    signals: list[int] = []
    armed = True
    for i, v in enumerate(vals):
        if math.isnan(v):
            continue
        if armed and v <= threshold:
            signals.append(i)
            armed = False
        elif not armed and v > rearm:
            armed = True
    return signals


def _excursion(ref, start, stop, low, high, close, mode):
    """Scan forward from `start` for the first recovery, measuring the low.

    mode 'touch': recovery the moment price trades back up to `ref` (high>=ref).
    mode 'close': recovery only when a bar *closes* back above `ref`.

    'touch' is the literal reading of the question, but on any noisy series the
    next bar's upper wick usually clips the prior close, so it collapses toward
    a one-bar measurement. 'close' asks for an actual reversal to hold.
    """
    rec = None
    j = start
    while j <= stop:
        hit = high[j] >= ref if mode == "touch" else close[j] >= ref
        if hit:
            rec = j
            break
        j += 1

    end = rec if rec is not None else stop
    seg = low[start : end + 1]
    low_idx = start + int(seg.argmin())
    # If price never traded below the reference the drop is zero, not negative:
    # the setup simply never went against you.
    drop = max(0.0, ref - float(seg.min()))
    return {
        "drop": drop,
        "low": float(seg.min()),
        "low_idx": low_idx,
        "rec_idx": rec,
        "recovered": rec is not None,
    }


def measure_events(
    df: pd.DataFrame,
    mfi: pd.Series,
    symbol: str,
    threshold: float,
    rearm: float,
    reference: str = "signal_close",
    include_signal_bar: bool = False,
    max_lookahead: int | None = None,
    recovery: str = "touch",
) -> pd.DataFrame:
    """Measure the downside excursion following every MFI signal."""
    contract = contract_for(symbol)
    pv = contract.point_value if contract else float("nan")
    mpv = contract.micro_point_value if contract else float("nan")

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    times = df.index
    mfi_vals = mfi.to_numpy(dtype=float)
    n = len(df)

    rows: list[Event] = []
    for i in find_signals(mfi, threshold, rearm):
        # The signal is only *known* once the bar closes, so by default the
        # excursion is measured over subsequent bars. --include-signal-bar
        # folds in the signal bar's own low for a chart-reading view.
        if reference == "signal_close":
            ref = close[i]
            start = i if include_signal_bar else i + 1
        elif reference == "next_open":
            if i + 1 >= n:
                continue
            ref = open_[i + 1]
            start = i + 1
        elif reference == "signal_low":
            ref = low[i]
            start = i + 1
        else:
            raise ValueError(f"unknown reference {reference!r}")

        if start >= n:
            continue

        stop = n - 1 if max_lookahead is None else min(n - 1, i + max_lookahead)

        touch = _excursion(ref, start, stop, low, high, close, "touch")
        closed = _excursion(ref, start, stop, low, high, close, "close")
        primary = touch if recovery == "touch" else closed

        if primary["recovered"]:
            censor = ""
        elif stop < n - 1:
            censor = f"no recovery within {max_lookahead} bars"
        else:
            censor = "still unrecovered at end of data"

        end = primary["rec_idx"] if primary["recovered"] else stop
        after = mfi_vals[i : end + 1]
        after = after[~np.isnan(after)]

        rows.append(Event(
            symbol=symbol,
            signal_time=times[i],
            signal_index=i,
            mfi_at_signal=float(mfi_vals[i]),
            min_mfi_after=float(after.min()) if after.size else float("nan"),
            reference_price=float(ref),
            lowest_low=primary["low"],
            lowest_low_time=times[primary["low_idx"]],
            points_dropped=primary["drop"],
            pct_dropped=float(100.0 * primary["drop"] / ref) if ref else float("nan"),
            dollars_dropped_full=float(primary["drop"] * pv),
            dollars_dropped_micro=float(primary["drop"] * mpv),
            bars_to_low=int(primary["low_idx"] - i),
            bars_to_recovery=int(primary["rec_idx"] - i) if primary["recovered"] else -1,
            hours_to_recovery=(
                float((times[primary["rec_idx"]] - times[i]).total_seconds() / 3600.0)
                if primary["recovered"] else float("nan")
            ),
            recovery_time=times[primary["rec_idx"]] if primary["recovered"] else None,
            recovered=primary["recovered"],
            censored_reason=censor,
            points_dropped_touch=touch["drop"],
            bars_to_recovery_touch=int(touch["rec_idx"] - i) if touch["recovered"] else -1,
            recovered_touch=touch["recovered"],
            points_dropped_close=closed["drop"],
            bars_to_recovery_close=int(closed["rec_idx"] - i) if closed["recovered"] else -1,
            recovered_close=closed["recovered"],
        ))

    return pd.DataFrame([asdict(r) for r in rows])


def compare_mfi(computed: pd.Series, reference: pd.Series) -> dict:
    """Cross-check our MFI against one exported from a charting platform.

    Any disagreement beyond floating-point noise means the two are not looking
    at the same thing -- usually a different MFI length, a different session
    setting (RTH vs ETH changes the volume), or bars anchored differently.
    """
    both = computed.notna() & reference.notna()
    n = int(both.sum())
    if n == 0:
        return {"compared": 0}
    diff = (computed[both] - reference[both]).abs()
    return {
        "compared": n,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "pct_within_0_01": float(100.0 * (diff <= 0.01).mean()),
        "pct_within_0_5": float(100.0 * (diff <= 0.5).mean()),
        "correlation": float(computed[both].corr(reference[both])),
    }


def format_mfi_check(label: str, res: dict) -> str:
    if res.get("compared", 0) == 0:
        return f"[{label}] no overlapping MFI values to compare."
    verdict = (
        "MATCH" if res["max_abs_diff"] <= 0.01
        else "CLOSE" if res["pct_within_0_5"] >= 99.0
        else "MISMATCH"
    )
    msg = (
        f"[{label}] MFI cross-check vs exported column: {verdict}  "
        f"(n={res['compared']}, max diff {res['max_abs_diff']:.4f}, "
        f"mean diff {res['mean_abs_diff']:.4f}, "
        f"{res['pct_within_0_01']:.1f}% within 0.01)"
    )
    if verdict == "MISMATCH":
        msg += (
            "\n    The two are not computing the same thing. Check that the "
            "chart's MFI length is 14,\n    that the chart session is ETH "
            "(RTH changes the volume), and that you exported 4h bars\n"
            "    and passed --no-resample."
        )
    return msg


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def summarise(events: pd.DataFrame, symbol: str) -> dict:
    contract = contract_for(symbol)
    done = events[events["recovered"]]
    pts = done["points_dropped"]

    def q(p: float) -> float:
        return float(pts.quantile(p)) if len(pts) else float("nan")

    summary = {
        "symbol": symbol,
        "contract": contract.name if contract else "unknown",
        "point_value_usd": contract.point_value if contract else float("nan"),
        "signals": int(len(events)),
        "resolved": int(len(done)),
        "unresolved": int((~events["recovered"]).sum()),
        "pts_mean": float(pts.mean()) if len(pts) else float("nan"),
        "pts_median": q(0.50),
        "pts_p25": q(0.25),
        "pts_p75": q(0.75),
        "pts_p90": q(0.90),
        "pts_p95": q(0.95),
        "pts_max": float(pts.max()) if len(pts) else float("nan"),
        "pct_median": float(done["pct_dropped"].median()) if len(done) else float("nan"),
        "pct_p90": float(done["pct_dropped"].quantile(0.90)) if len(done) else float("nan"),
        "no_adverse_move_pct": float(100.0 * (pts == 0).mean()) if len(pts) else float("nan"),
        "bars_to_recovery_median": float(done["bars_to_recovery"].median()) if len(done) else float("nan"),
        "bars_to_recovery_p90": float(done["bars_to_recovery"].quantile(0.90)) if len(done) else float("nan"),
        "bars_to_low_median": float(done["bars_to_low"].median()) if len(done) else float("nan"),
    }

    # The same signals measured under the stricter close-through rule.
    dc = events[events["recovered_close"]]
    summary["close_rule_resolved"] = int(len(dc))
    summary["close_pts_median"] = float(dc["points_dropped_close"].median()) if len(dc) else float("nan")
    summary["close_pts_p90"] = float(dc["points_dropped_close"].quantile(0.90)) if len(dc) else float("nan")
    summary["close_pts_max"] = float(dc["points_dropped_close"].max()) if len(dc) else float("nan")
    summary["close_bars_median"] = float(dc["bars_to_recovery_close"].median()) if len(dc) else float("nan")
    dt = events[events["recovered_touch"]]
    summary["touch_pts_median"] = float(dt["points_dropped_touch"].median()) if len(dt) else float("nan")
    summary["touch_pts_p90"] = float(dt["points_dropped_touch"].quantile(0.90)) if len(dt) else float("nan")
    summary["touch_bars_median"] = float(dt["bars_to_recovery_touch"].median()) if len(dt) else float("nan")
    summary["touch_resolved_next_bar_pct"] = (
        float(100.0 * (dt["bars_to_recovery_touch"] <= 1).mean()) if len(dt) else float("nan")
    )
    if contract:
        summary["usd_median_full"] = summary["pts_median"] * contract.point_value
        summary["usd_p90_full"] = summary["pts_p90"] * contract.point_value
        summary["usd_median_micro"] = summary["pts_median"] * contract.micro_point_value
        summary["usd_p90_micro"] = summary["pts_p90"] * contract.micro_point_value
        summary["close_usd_median_full"] = summary["close_pts_median"] * contract.point_value
        summary["close_usd_p90_full"] = summary["close_pts_p90"] * contract.point_value
    return summary


def stop_survival(events: pd.DataFrame, levels: list[float]) -> pd.DataFrame:
    """For each candidate stop distance, what fraction of signals survived it?

    This is the practical read on the question: 'if I bought when MFI hit 20
    and placed my stop N points away, how often would I have been stopped
    before price came back?'
    """
    done = events[events["recovered"]]
    rows = []
    for lv in levels:
        if len(done) == 0:
            rows.append({"stop_points": lv, "survived_pct": float("nan"), "stopped_out": 0})
            continue
        survived = (done["points_dropped"] <= lv).mean() * 100.0
        rows.append({
            "stop_points": lv,
            "survived_pct": float(survived),
            "stopped_out": int((done["points_dropped"] > lv).sum()),
        })
    return pd.DataFrame(rows)


def default_stop_levels(symbol: str, events: pd.DataFrame) -> list[float]:
    pts = events.loc[events["recovered"], "points_dropped"]
    if not len(pts):
        return []
    top = float(pts.quantile(0.95))
    step = max(round(top / 8.0, 1), 0.1)
    return [round(step * k, 1) for k in range(1, 9)]


def fmt_table(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:,.2f}")


def report(all_events: dict[str, pd.DataFrame], args) -> str:
    out = []
    out.append("=" * 78)
    out.append("MFI-20 DOWNSIDE EXCURSION STUDY")
    out.append("=" * 78)
    out.append(
        f"timeframe={args.timeframe}  mfi_length={args.mfi_length}  "
        f"threshold={args.threshold}  re-arm above={args.rearm}  "
        f"reference={args.reference}  recovery={args.recovery}  tz={args.tz}"
    )
    if getattr(args, "mfi_checks", None):
        out.append("")
        for line in args.mfi_checks:
            out.append(line)
    out.append("")
    out.append("Measured: points from the reference price down to the lowest low")
    out.append("printed before the first bar trading back up through that price.")
    out.append("")

    summaries = [summarise(ev, sym) for sym, ev in all_events.items() if len(ev)]
    if summaries:
        s_df = pd.DataFrame(summaries)
        out.append("-" * 78)
        out.append("HEADLINE  (points of drawdown before recovery)")
        out.append("-" * 78)
        cols = ["symbol", "signals", "resolved", "unresolved", "pts_median",
                "pts_mean", "pts_p75", "pts_p90", "pts_p95", "pts_max"]
        out.append(fmt_table(s_df[cols]))
        out.append("")
        out.append("-" * 78)
        out.append("SAME NUMBERS IN DOLLARS AND PERCENT")
        out.append("-" * 78)
        cols = ["symbol", "point_value_usd", "usd_median_full", "usd_p90_full",
                "usd_median_micro", "usd_p90_micro", "pct_median", "pct_p90"]
        out.append(fmt_table(s_df[[c for c in cols if c in s_df.columns]]))
        out.append("")
        out.append("-" * 78)
        out.append("TOUCH-BACK vs CLOSE-THROUGH  (why the definition matters)")
        out.append("-" * 78)
        out.append("  touch_*: recovery the instant price trades back to the reference")
        out.append("  close_*: recovery only once a 4h bar CLOSES back above it")
        cols = ["symbol", "touch_pts_median", "touch_pts_p90", "touch_bars_median",
                "touch_resolved_next_bar_pct", "close_pts_median", "close_pts_p90",
                "close_pts_max", "close_bars_median"]
        out.append(fmt_table(s_df[[c for c in cols if c in s_df.columns]]))
        out.append("")
        out.append("-" * 78)
        out.append("TIMING  (4h bars from signal)")
        out.append("-" * 78)
        cols = ["symbol", "bars_to_low_median", "bars_to_recovery_median",
                "bars_to_recovery_p90", "no_adverse_move_pct"]
        out.append(fmt_table(s_df[cols]))
        out.append("")

    for sym, ev in all_events.items():
        if not len(ev):
            out.append(f"[{sym}] no signals in the sample.")
            continue
        out.append("-" * 78)
        out.append(f"STOP SURVIVAL — {sym}")
        out.append("-" * 78)
        levels = args.stop_levels or default_stop_levels(sym, ev)
        out.append(fmt_table(stop_survival(ev, levels)))
        unres = ev[~ev["recovered"]]
        if len(unres):
            out.append("")
            out.append(f"  {len(unres)} unresolved signal(s):")
            for _, r in unres.iterrows():
                out.append(
                    f"    {r['signal_time']}  ref {r['reference_price']:,.2f}  "
                    f"low {r['lowest_low']:,.2f}  "
                    f"open drawdown {r['points_dropped']:,.2f} pts  "
                    f"({r['censored_reason']})"
                )
        out.append("")

    out.append("-" * 78)
    out.append("READ THIS BEFORE TRADING THE NUMBERS")
    out.append("-" * 78)
    out.append(
        "* Unresolved signals are excluded from the percentiles. They are the\n"
        "  worst cases by definition, so every statistic below is optimistic\n"
        "  whenever the unresolved count is non-zero.\n"
        "* Continuous/back-adjusted futures series contain roll gaps. A roll\n"
        "  inside an open excursion distorts that event; front-month-only data\n"
        "  or a roll-aware series is cleaner.\n"
        "* MFI is volume-weighted. Index-CFD or spot-gold feeds carry synthetic\n"
        "  volume and will not reproduce exchange-volume MFI readings.\n"
        "* Bar anchoring changes the answer. These 4h bars are anchored to the\n"
        "  18:00 New York session open, matching how charting platforms bucket\n"
        "  CME/COMEX futures."
    )
    return "\n".join(out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_pairs(items: list[str]) -> dict[str, str]:
    """Parse LABEL=VALUE, tolerating values that themselves contain '=' (NQ=F)."""
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"Expected LABEL=VALUE, got {it!r}")
        label, value = it.split("=", 1)
        out[label.strip()] = value.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("data source")
    src.add_argument("--csv", nargs="+", metavar="LABEL=PATH",
                     help="One or more CSVs, e.g. NQ=nq_1h.csv ES=es_1h.csv")
    src.add_argument("--yahoo", nargs="+", metavar="LABEL=SYMBOL",
                     help="Yahoo symbols, e.g. NQ=NQ=F (needs network access)")
    src.add_argument("--databento", nargs="+", metavar="LABEL=SPEC",
                     help="Databento roots, e.g. NQ=NQ ES=ES GC=GC, or an "
                          "explicit LABEL=DATASET:SYMBOL:STYPE. Reads the key "
                          "from $DATABENTO_API_KEY.")
    src.add_argument("--start", default="2020-01-01", help="Databento start date")
    src.add_argument("--end", default=None, help="Databento end date (default: now)")
    src.add_argument("--period", default="730d", help="Yahoo lookback (default 730d)")
    src.add_argument("--interval", default="1h", help="Yahoo base interval (default 1h)")

    sh = p.add_argument_group("bar construction")
    sh.add_argument("--timeframe", default="4h", help="Target bar size (default 4h)")
    sh.add_argument("--no-resample", action="store_true",
                    help="Input is already on the target timeframe")
    sh.add_argument("--tz", default="America/New_York", help="Session timezone")
    sh.add_argument("--session-start-hour", type=int, default=18,
                    help="Hour that anchors the bars (default 18 = CME open)")

    ind = p.add_argument_group("signal definition")
    ind.add_argument("--mfi-length", type=int, default=14)
    ind.add_argument("--threshold", type=float, default=20.0)
    ind.add_argument("--rearm", type=float, default=None,
                     help="MFI must close back above this before a new signal "
                          "counts (default: same as --threshold)")
    ind.add_argument("--reference", default="signal_close",
                     choices=["signal_close", "next_open", "signal_low"],
                     help="Price the recovery is measured against")
    ind.add_argument("--include-signal-bar", action="store_true",
                     help="Count the signal bar's own low in the excursion")
    ind.add_argument("--recovery", default="touch", choices=["touch", "close"],
                     help="touch: price trades back to the reference (literal). "
                          "close: a bar must CLOSE back above it (default: touch)")
    ind.add_argument("--max-lookahead", type=int, default=None,
                     help="Give up on recovery after this many bars")

    out = p.add_argument_group("output")
    out.add_argument("--outdir", default="out")
    out.add_argument("--stop-levels", type=float, nargs="+", default=None,
                     help="Explicit stop distances for the survival table")
    out.add_argument("--volume-proxy", action="store_true",
                     help="Substitute constant volume when the feed has none "
                          "(changes what MFI measures — see README)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.rearm is None:
        args.rearm = args.threshold
    if not args.csv and not args.yahoo and not args.databento:
        build_parser().print_help()
        return 2

    sources: dict[str, pd.DataFrame] = {}
    if args.csv:
        for label, path in parse_pairs(args.csv).items():
            if not os.path.exists(path):
                raise SystemExit(f"No such file: {path}")
            sources[label] = load_csv(path, args.tz)
    if args.yahoo:
        for label, sym in parse_pairs(args.yahoo).items():
            sources[label] = load_yahoo(sym, args.period, args.interval, args.tz)
    if args.databento:
        for label, spec in parse_pairs(args.databento).items():
            print(f"[{label}] fetching {spec} from Databento ...", file=sys.stderr)
            sources[label] = load_databento(
                spec, args.start, args.end, args.interval, args.tz)

    os.makedirs(args.outdir, exist_ok=True)
    all_events: dict[str, pd.DataFrame] = {}
    mfi_checks: list[str] = []

    for label, raw in sources.items():
        bars = raw if args.no_resample else resample_bars(
            raw, args.timeframe, args.session_start_hour)

        if args.volume_proxy and float(bars["volume"].sum()) == 0.0:
            bars = bars.assign(volume=1.0)
        if float(bars["volume"].sum()) == 0.0:
            raise SystemExit(
                f"[{label}] all volume is zero — MFI cannot be computed. "
                "Use a feed with real volume, or pass --volume-proxy."
            )

        mfi = money_flow_index(bars, args.mfi_length)

        if "mfi_ref" in bars.columns:
            check = compare_mfi(mfi, bars["mfi_ref"])
            mfi_checks.append(format_mfi_check(label, check))
            print(mfi_checks[-1], file=sys.stderr)
        elif not args.no_resample:
            pass
        events = measure_events(
            bars, mfi, label,
            threshold=args.threshold,
            rearm=args.rearm,
            reference=args.reference,
            include_signal_bar=args.include_signal_bar,
            max_lookahead=args.max_lookahead,
            recovery=args.recovery,
        )
        all_events[label] = events

        span = f"{bars.index[0]:%Y-%m-%d} .. {bars.index[-1]:%Y-%m-%d}"
        print(f"[{label}] {len(bars)} {args.timeframe} bars  ({span})  "
              f"-> {len(events)} signals", file=sys.stderr)

        events.to_csv(os.path.join(args.outdir, f"{label}_events.csv"), index=False)
        bars.assign(mfi=mfi).to_csv(os.path.join(args.outdir, f"{label}_bars.csv"))

    args.mfi_checks = mfi_checks
    text = report(all_events, args)
    print(text)
    with open(os.path.join(args.outdir, "report.txt"), "w") as fh:
        fh.write(text + "\n")

    combined = [ev for ev in all_events.values() if len(ev)]
    if combined:
        pd.concat(combined).to_csv(
            os.path.join(args.outdir, "all_events.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
