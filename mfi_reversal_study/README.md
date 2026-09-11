# MFI-20 downside excursion study (NQ / ES / GC, 4h)

**Question:** on a 4-hour chart, every time MFI reaches 20, how many points does
price drop before it reverses back up through the price it was at when MFI first
hit 20?

For every signal the tool records the **Maximum Adverse Excursion**:

| term | definition |
| --- | --- |
| signal | the 4h bar on which MFI(14) first closes at or below 20 |
| reference price | the close of that signal bar |
| recovery | the first later bar that gets back up to the reference price |
| **points dropped** | reference price − lowest low printed before recovery |

A signal only re-arms once MFI closes back **above** 20, so an extended stretch
of MFI pinned under 20 counts as one event rather than a dozen.

## Run it

```bash
pip install pandas numpy databento

export DATABENTO_API_KEY='db-...'        # never pass the key as an argument
python mfi_study.py --databento NQ=NQ ES=ES GC=GC \
                    --start 2019-01-01 \
                    --outdir out
```

`NQ`/`ES`/`GC` expand to the CME Globex front-month continuous contracts
(`NQ.c.0`, `ES.c.0`, `GC.c.0` on `GLBX.MDP3`). Hourly bars are fetched and
aggregated to 4h. For an explicit contract use `LABEL=DATASET:SYMBOL:STYPE`,
e.g. `NQ=GLBX.MDP3:NQZ5:raw_symbol`.

### From TradingView exports

Set the chart up before exporting — these three settings decide whether the
numbers mean anything:

1. **Symbol:** `CME_MINI:NQ1!`, `CME_MINI:ES1!`, `COMEX:GC1!` (front-month
   continuous).
2. **Interval: 4h**, and set the session to **extended hours (ETH)**. RTH-only
   changes the volume, and MFI is volume-weighted, so an RTH chart produces a
   different indicator.
3. **Add two studies to the chart: `Volume` and `Money Flow Index` (length 14).**
   TradingView only exports indicators that are on the chart. Volume is
   required — MFI cannot be computed without it. The MFI column is optional but
   worth adding: the tool cross-checks its own MFI against it (see below).

Then `⋯` menu (top right of the chart) → **Export chart data…** → time format
ISO, and run:

```bash
python mfi_study.py --csv NQ=nq_4h.csv ES=es_4h.csv GC=gc_4h.csv \
                    --no-resample --outdir out
```

`--no-resample` matters: the bars are already 4h and already anchored the way
TradingView anchors them, so re-bucketing them would be wrong.

**The cross-check.** If the export contains an MFI column, the tool compares it
against its own calculation and prints one of:

```
[NQ] MFI cross-check vs exported column: MATCH  (n=2886, max diff 0.0001, ...)
[NQ] MFI cross-check vs exported column: MISMATCH (n=2880, max diff 57.50, ...)
```

`MATCH` means this tool and your chart agree bar for bar, so the signals it
finds are the ones you would see on screen. `MISMATCH` means they are not
computing the same thing — wrong MFI length, RTH instead of ETH, or bars that
got resampled when they should not have been.

### From other vendors

Column names and timestamp formats are auto-detected (epoch seconds/millis/
micros/nanos, or ISO with or without an offset) for NinjaTrader, Barchart,
IBKR, Databento and similar:

```bash
python mfi_study.py --csv NQ=nq_1h.csv ES=es_1h.csv GC=gc_1h.csv --outdir out
```

## The one decision you should make consciously

"Reversing upward past the price" has two defensible readings, and they give
very different answers:

- **`--recovery touch`** (default) — recovery the instant price *trades* back to
  the reference. This is the literal reading, but on any noisy series the next
  bar's upper wick usually clips the prior close, so almost every signal
  "recovers" on the very next bar and the statistic collapses into a one-bar
  measurement.
- **`--recovery close`** — recovery only once a 4h bar *closes* back above the
  reference. This demands the reversal actually hold.

Both are computed on every run regardless of the flag, and the
**TOUCH-BACK vs CLOSE-THROUGH** table in the report puts them side by side. Look
at `touch_resolved_next_bar_pct`: if it is near 100%, the touch numbers are
measuring noise, not a reversal.

## Output

Written to `--outdir`:

- `report.txt` — headline percentiles in points, dollars and percent; the
  touch-vs-close comparison; recovery timing; and a **stop-survival** table
  answering "with a stop N points away, how often would I have survived?"
- `<SYM>_events.csv` — one row per signal, both recovery rules, timestamps
- `<SYM>_bars.csv` — the resampled 4h bars with the MFI series, for charting

Point values used: NQ $20, ES $50, GC $100 (micros: $2 / $5 / $10).

## Options that change the answer

| flag | default | effect |
| --- | --- | --- |
| `--mfi-length` | 14 | MFI lookback |
| `--threshold` | 20 | oversold trigger |
| `--rearm` | = threshold | MFI must close above this before a new signal counts |
| `--recovery` | `touch` | `touch` or `close`, as above |
| `--reference` | `signal_close` | or `next_open` (tradeable fill) / `signal_low` |
| `--include-signal-bar` | off | count the signal bar's own low in the excursion |
| `--max-lookahead` | none | abandon a signal after N bars instead of running to the end of data |
| `--session-start-hour` | 18 | bar anchoring, New York time |

## Caveats that materially affect the numbers

- **Unresolved signals are excluded from the percentiles.** They are the worst
  cases by definition, so every statistic is optimistic whenever the
  `unresolved` count is non-zero. Check that column first.
- **Bar anchoring changes the signals.** 4h bars are anchored to the 18:00 New
  York futures open, matching how charting platforms bucket CME/COMEX futures.
  Anchoring at midnight produces different bars, a different MFI series and
  therefore different events.
- **Continuous contracts contain roll gaps.** A roll inside an open excursion
  distorts that event. `.c.0` is front-month-continuous without back-adjustment;
  for a clean read, run single contracts.
- **MFI is volume-weighted.** CFD or spot-gold feeds carry synthetic volume and
  will not reproduce exchange-volume MFI.
- This measures what price *did* after a signal. It is not a strategy: there is
  no entry rule, no exit rule and no costs.

## Development

```bash
python test_mfi_study.py     # 56 assertions, no pytest needed
python make_demo_data.py     # synthetic bars for offline smoke-testing only
```

The test suite checks MFI against a hand-computed value and an independent
unvectorised implementation, and pins the excursion logic against hand-built
cases (post-recovery lows excluded, no-adverse-move clamped to zero, unrecovered
signals flagged rather than dropped, close-rule ≥ touch-rule on every event).

Synthetic demo data exists purely to exercise the pipeline — its statistics say
nothing about real market behaviour.
