# =============================================================================
# crypto_scanner.py — Crypto Pivot + EMA Scanner
# Binance Spot | Top 50 USDT pairs | 1H TF | WhatsApp + Telegram Alerts
#
# Signal Logic:
#
# RETEST SELL:
#   Step 1 → Price closes BELOW any pivot (break first)
#   Step 2 → EMA 9 crosses BELOW EMA 26 (after break)
#   Step 3 → Within next 5 candles: High touches EMA9 BUT Close < EMA9
#   → 🔴 SELL RETEST ALERT
#
# AGGRESSIVE SELL:
#   Step 1 → EMA 9 crosses BELOW EMA 26 (crossover first)
#   Step 2 → Price closes BELOW any pivot (after crossover)
#   → 🔴 SELL AGGRESSIVE ALERT
#
# RETEST BUY:
#   Step 1 → Price closes ABOVE any pivot (break first)
#   Step 2 → EMA 9 crosses ABOVE EMA 26 (after break)
#   Step 3 → Within next 5 candles: Low touches EMA9 BUT Close > EMA9
#   → 🟢 BUY RETEST ALERT
#
# AGGRESSIVE BUY:
#   Step 1 → EMA 9 crosses ABOVE EMA 26 (crossover first)
#   Step 2 → Price closes ABOVE any pivot (after crossover)
#   → 🟢 BUY AGGRESSIVE ALERT
# SNIPER BUY:
#   1H: 1H EMA9 must be ABOVE EMA26 at trigger (bullish 1H alignment gate)
#   1H: Price consolidates within 3.5% below pivot for min 5 candles
#   1H: EMA 9 slants upward during consolidation (flat → rising)
#   1H: Candle closes ABOVE pivot AND above EMA 9
#   1H: At least 40% of candle body must close above pivot
#   30m: EMA 9 above EMA 26 (alignment check — not crossover hunting)
#   → 🟢 BUY SNIPER ALERT
#
# SNIPER SELL:
#   1H: 1H EMA9 must be BELOW EMA26 at trigger (bearish 1H alignment gate)
#   1H: Price consolidates within 3.5% above pivot for min 5 candles
#   1H: EMA 9 slants downward during consolidation (flat → falling)
#   1H: Candle closes BELOW pivot AND below EMA 9
#   1H: At least 40% of candle body must close below pivot
#   30m: EMA 9 below EMA 26 (alignment check — not crossover hunting)
#   → 🔴 SELL SNIPER ALERT
# IMPULSE BUY:
#   Leg 1  : 3-5 candles sharp upward move, max 1 small red candle
#            EMA9 > EMA26 throughout
#   Consol : 4-10 tight candles near top of leg 1 (upper 60% of leg range)
#            EMA9 > EMA26 maintained, avg body smaller than leg bodies
#   Trigger: single green candle whose BODY engulfs full consolidation range
#            EMA9 > EMA26 at trigger
#   → 🟢 BUY IMPULSE ALERT
#
# IMPULSE SELL:
#   Leg 1  : 3-5 candles sharp downward move, max 1 small green candle
#            EMA9 < EMA26 throughout
#   Consol : 4-10 tight candles near bottom of leg 1 (lower 60% of leg range)
#            EMA9 < EMA26 maintained, avg body smaller than leg bodies
#   Trigger: single red candle whose BODY engulfs full consolidation range
#            EMA9 < EMA26 at trigger
#   → 🔴 SELL IMPULSE ALERT
# =============================================================================
import pandas as pd
import numpy as np
import time
import os
import schedule
from datetime import datetime, timezone
from twilio.rest import Client as TwilioClient
from collections import defaultdict
from crypto_pivot import get_pivots_for_symbol
from crypto_config import *

# ─────────────────────────────────────────────────────────────────────────────
# STATE TRACKING PER COIN
# ─────────────────────────────────────────────────────────────────────────────
coin_state   = defaultdict(dict)   # State machine per coin
DELTA_BASE_URL = "https://api.india.delta.exchange"
alerted_this_cycle = set()         # Avoid duplicate alerts same scan cycle
weekly_pivots = {}                 # Cache pivots for the week

# ─────────────────────────────────────────────────────────────────────────────
# BINANCE DATA
# ─────────────────────────────────────────────────────────────────────────────
def get_top_50_symbols():
    """
    Fetch top N perpetual futures by 24h turnover from Delta Exchange India.
    Uses /v2/tickers which includes volume info per product.
    """
    try:
        url      = f"{DELTA_BASE_URL}/v2/tickers"
        response = requests.get(url, timeout=10)
        data     = response.json()

        if not data.get("success"):
            raise Exception(f"API returned success=false: {data}")

        tickers = data.get("result", [])

        # Filter: USD-margined perpetual futures only, live state
        usd_perps = [
            t for t in tickers
            if t.get("symbol", "").endswith("USD")
            and not t.get("symbol", "").endswith("USDT")  # exclude any USDT pairs if present
            and t.get("contract_type") == "perpetual_futures"
        ]

        # Sort by 24h turnover in USD — confirmed field name via live API test
        # on JTOUSD (2026-06-19): turnover_usd == turnover == 467959.29,
        # turnover_symbol == "USD". Using turnover_usd explicitly since the
        # field name itself documents its unit, removing ambiguity.
        def get_vol(t):
            val = t.get("turnover_usd")
            try:
                return float(val) if val is not None else 0.0
            except (ValueError, TypeError):
                return 0.0

        usd_perps.sort(key=get_vol, reverse=True)

        symbols = [t["symbol"] for t in usd_perps[:TOP_N_COINS]]
        print(f"  ✅ Top {len(symbols)} Delta perpetuals by 24h volume fetched")
        if symbols:
            print(f"  Sample: {symbols[:5]}")
        return symbols

    except Exception as e:
        print(f"  ❌ Error fetching symbols from Delta: {e}")
        print(f"  Falling back to hardcoded list — VERIFY these symbols exist on Delta India.")
        return [
            "BTCUSD","ETHUSD","SOLUSD","XRPUSD","DOGEUSD",
            "ADAUSD","AVAXUSD","DOTUSD","LINKUSD","LTCUSD",
            "BCHUSD","NEARUSD","ARBUSD","OPUSD","SUIUSD",
            "APTUSD","STXUSD","ICPUSD","TRXUSD","PEPEUSD"
        ]


def get_1h_candles(symbol, limit=100):
    """
    Fetch last N closed 1H candles from Delta Exchange India.
    Returns DataFrame with columns: open, high, low, close, volume
    """
    try:
        end_ts   = int(datetime.now(timezone.utc).timestamp())
        start_ts = end_ts - (limit + 5) * 3600  # +5 buffer candles

        url    = f"{DELTA_BASE_URL}/v2/history/candles"
        params = {
            "symbol"    : symbol,
            "resolution": "1h",
            "start"     : start_ts,
            "end"       : end_ts
        }
        response = requests.get(url, params=params, timeout=10)
        data     = response.json()

        if not data.get("success") or not data.get("result"):
            return None

        candles = data["result"]  # newest first, per confirmed test on JTOUSD

        df = pd.DataFrame(candles)
        if df.empty:
            return None

        df = df.astype({
            "open" : float, "high": float,
            "low"  : float, "close": float,
            "volume": float
        })

        # Sort oldest -> newest (Delta returns newest first)
        df = df.sort_values("time").reset_index(drop=True)

        # Drop the last candle if it's still the currently-forming hour
        now_hour_ts = int(datetime.now(timezone.utc).timestamp() // 3600 * 3600)
        if len(df) > 0 and df.iloc[-1]["time"] >= now_hour_ts:
            df = df.iloc[:-1].reset_index(drop=True)

        return df

    except Exception as e:
        return None


def get_30m_candles(symbol, limit=60):
    """
    Fetch last N closed 30min candles from Delta Exchange India.
    Used exclusively for SNIPER signal — 30min EMA 9/26 crossover check.
    Returns DataFrame with columns: open, high, low, close, volume, time
    """
    try:
        end_ts   = int(datetime.now(timezone.utc).timestamp())
        start_ts = end_ts - (limit + 5) * 1800  # +5 buffer candles

        url    = f"{DELTA_BASE_URL}/v2/history/candles"
        params = {
            "symbol"    : symbol,
            "resolution": "30m",
            "start"     : start_ts,
            "end"       : end_ts
        }
        response = requests.get(url, params=params, timeout=10)
        data     = response.json()

        if not data.get("success") or not data.get("result"):
            return None

        candles = data["result"]
        df = pd.DataFrame(candles)
        if df.empty:
            return None

        df = df.astype({
            "open": float, "high": float,
            "low" : float, "close": float,
            "volume": float
        })

        # Sort oldest -> newest
        df = df.sort_values("time").reset_index(drop=True)

        # Drop the currently-forming 30min candle
        now_30m_ts = int(datetime.now(timezone.utc).timestamp() // 1800 * 1800)
        if len(df) > 0 and df.iloc[-1]["time"] >= now_30m_ts:
            df = df.iloc[:-1].reset_index(drop=True)

        return df

    except Exception as e:
        return None


def check_30m_ema_alignment(symbol, direction):
    """
    SNIPER helper — checks if 30min EMA 9/26 alignment matches the trade direction.

    direction = "BUY"  → EMA9 must be ABOVE EMA26 on the latest closed 30min candle
    direction = "SELL" → EMA9 must be BELOW EMA26 on the latest closed 30min candle

    No crossover event hunting — just current state of EMAs on 30min TF.
    Returns True if aligned, False otherwise.
    """
    try:
        df30 = get_30m_candles(symbol, limit=60)
        if df30 is None or len(df30) < 30:
            return False

        df30["ema9"]  = df30["close"].ewm(span=9,  adjust=False).mean()
        df30["ema26"] = df30["close"].ewm(span=26, adjust=False).mean()

        latest = df30.iloc[-1]

        if direction == "BUY":
            return latest["ema9"] > latest["ema26"]
        else:
            return latest["ema9"] < latest["ema26"]

    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# EMA CALCULATION
# ─────────────────────────────────────────────────────────────────────────────
def calculate_ema(series, period):
    """
    Calculate EMA — same as TradingView Pine Script ta.ema()
    Uses Wilder's smoothing: multiplier = 2 / (period + 1)
    """
    return series.ewm(span=period, adjust=False).mean()


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_signals(symbol, df, pivots):
    """
    Signal detection — sequential state machine with chop filtering and
    crossover-reuse prevention.

    RETEST path (per pivot level, per direction):
        idle    → waiting for a clean break (see chop filter below)
        broken  → break confirmed, waiting for EMA crossover (max MAX_WAIT_FOR_CROSS candles)
        crossed → crossover confirmed, waiting for EMA9 retest (max RETEST_CANDLES candles)
        State resets fully if price closes back through the pivot, or if
        either waiting window expires.

    AGGRESSIVE path (per direction, global per symbol — not per pivot level):
        Tracks the index of the LAST EMA crossover already used to fire an
        AGGRESSIVE signal (per direction). A crossover can only ever produce
        ONE aggressive signal, at the first pivot level price closes beyond
        after that crossover. Once used, that crossover index is marked
        consumed and cannot trigger again, even if price re-crosses other
        pivot levels later using the same stale crossover via the lookback
        window. A NEW crossover is required for the next aggressive signal.

        This fixes a confirmed bug (2026-06-20, SOLUSD real data) where one
        EMA crossover at 19:00 produced repeat AGGRESSIVE signals at 04:00
        and 07:00 the next day, because the lookback window kept finding the
        same old crossover as "most recent" each time price chopped across
        a pivot with no fresh crossover involved.

    CHOP FILTER (applies to both paths):
        A "clean break" requires ALL THREE:
        1. Previous candle closed on the OPPOSITE side of the pivot
        2. Current candle closes beyond the pivot
        3. At least 25% of the candle BODY closes beyond the pivot
        Prevents signals on candles already sitting on the wrong side,
        or candles that only barely clip the pivot with a tiny wick/body.

    Timestamps: every signal is stamped with the ACTUAL CANDLE's time (from
    the candle's own "time" field, converted to UTC), not the wall-clock
    time the scan happened to run. This was a separate confirmed bug — all
    signals in earlier test runs were mis-stamped with script-run time.

    Only the latest closed candle (curr_idx) can trigger a signal, so each
    setup fires exactly once, at the candle where it completes.
    """
    if df is None or len(df) < EMA_SLOW + 10:
        return []

    if "time" not in df.columns:
        # Safety: signals need real candle timestamps to be trustworthy.
        # Refuse to guess — fail loudly instead of mis-stamping silently.
        raise ValueError(f"detect_signals({symbol}): DataFrame missing 'time' column — cannot stamp signals correctly.")

    signals  = []
    curr_idx = len(df) - 1

    df["ema9"]  = calculate_ema(df["close"], EMA_FAST)
    df["ema26"] = calculate_ema(df["close"], EMA_SLOW)
    df["bear_cross"] = (df["ema9"] < df["ema26"]) & (df["ema9"].shift(1) >= df["ema26"].shift(1))
    df["bull_cross"] = (df["ema9"] > df["ema26"]) & (df["ema9"].shift(1) <= df["ema26"].shift(1))

    def candle_time_str(idx):
        ts = df.iloc[idx]["time"]
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    pivot_levels = {
        "R3": pivots["R3"], "R2": pivots["R2"], "R1": pivots["R1"],
        "P" : pivots["P"],
        "S1": pivots["S1"], "S2": pivots["S2"], "S3": pivots["S3"]
    }

    MAX_WAIT_FOR_CROSS   = 15   # candles after break to wait for EMA crossover (RETEST path)
    MIN_BODY_BREAK_PCT   = 0.30 # at least 25% of candle body must close beyond pivot

    if (symbol not in coin_state
            or not isinstance(coin_state[symbol], dict)
            or "R3" not in coin_state[symbol]):
        coin_state[symbol] = {
            lvl: {
                "sell": {"phase": "idle", "break_idx": None, "cross_idx": None},
                "buy" : {"phase": "idle", "break_idx": None, "cross_idx": None},
                "_agg_bear_cross_used": None,
                "_agg_bull_cross_used": None,
            }
            for lvl in pivot_levels
        }
        coin_state[symbol]["_last_idx"] = EMA_SLOW

    state = coin_state[symbol]
    start_idx = max(state.get("_last_idx", EMA_SLOW), EMA_SLOW, 1)

    for i in range(start_idx, curr_idx + 1):
        candle    = df.iloc[i]
        is_latest = (i == curr_idx)

        for level_name, level_price in pivot_levels.items():
            sell_st = state[level_name]["sell"]
            buy_st  = state[level_name]["buy"]

            prev_close  = df.iloc[i - 1]["close"]
            body_size   = abs(candle["close"] - candle["open"])

            # SELL break: prev closed above pivot, current closes below
            # + at least 25% of body is below pivot
            if body_size > 0:
                body_below  = (level_price - candle["close"]) / body_size
                body_above  = (candle["close"] - level_price) / body_size
            else:
                body_below = body_above = 0.0

            clean_break_below = (
                prev_close  >= level_price and
                candle["close"] <  level_price and
                body_below  >= MIN_BODY_BREAK_PCT
            )
            clean_break_above = (
                prev_close  <= level_price and
                candle["close"] >  level_price and
                body_above  >= MIN_BODY_BREAK_PCT
            )

            # ── SELL state machine (RETEST path) ────────────────────────────
            if sell_st["phase"] == "idle":
                if clean_break_below:
                    sell_st["phase"], sell_st["break_idx"] = "broken", i

            elif sell_st["phase"] == "broken":
                if candle["close"] >= level_price:
                    sell_st["phase"], sell_st["break_idx"] = "idle", None
                elif candle["bear_cross"]:
                    sell_st["phase"], sell_st["cross_idx"] = "crossed", i
                elif i - sell_st["break_idx"] > MAX_WAIT_FOR_CROSS:
                    sell_st["phase"], sell_st["break_idx"] = "idle", None

            elif sell_st["phase"] == "crossed":
                if i - sell_st["cross_idx"] > RETEST_CANDLES:
                    sell_st["phase"], sell_st["break_idx"], sell_st["cross_idx"] = "idle", None, None
                elif candle["close"] >= level_price:
                    sell_st["phase"], sell_st["break_idx"], sell_st["cross_idx"] = "idle", None, None
                else:
                    ema9_i = candle["ema9"]
                    if candle["high"] >= ema9_i * (1 - EMA_TOUCH_BUFFER_PCT) and candle["close"] < ema9_i:
                        if is_latest:
                            sig_key = f"{symbol}_RETEST_SELL_{level_name}_{sell_st['break_idx']}"
                            if sig_key not in alerted_this_cycle:
                                signals.append({
                                    "symbol": symbol, "type": "SELL", "entry": "RETEST",
                                    "pivot_name": level_name, "pivot_price": level_price,
                                    "cmp": round(candle["close"], 6),
                                    "ema9": round(ema9_i, 6), "ema26": round(candle["ema26"], 6),
                                    "time": candle_time_str(i),
                                })
                                alerted_this_cycle.add(sig_key)
                        sell_st["phase"], sell_st["break_idx"], sell_st["cross_idx"] = "idle", None, None

            # ── BUY state machine (RETEST path) ─────────────────────────────
            if buy_st["phase"] == "idle":
                if clean_break_above:
                    buy_st["phase"], buy_st["break_idx"] = "broken", i

            elif buy_st["phase"] == "broken":
                if candle["close"] <= level_price:
                    buy_st["phase"], buy_st["break_idx"] = "idle", None
                elif candle["bull_cross"]:
                    buy_st["phase"], buy_st["cross_idx"] = "crossed", i
                elif i - buy_st["break_idx"] > MAX_WAIT_FOR_CROSS:
                    buy_st["phase"], buy_st["break_idx"] = "idle", None

            elif buy_st["phase"] == "crossed":
                if i - buy_st["cross_idx"] > RETEST_CANDLES:
                    buy_st["phase"], buy_st["break_idx"], buy_st["cross_idx"] = "idle", None, None
                elif candle["close"] <= level_price:
                    buy_st["phase"], buy_st["break_idx"], buy_st["cross_idx"] = "idle", None, None
                else:
                    ema9_i = candle["ema9"]
                    if candle["low"] <= ema9_i * (1 + EMA_TOUCH_BUFFER_PCT) and candle["close"] > ema9_i:
                        if is_latest:
                            sig_key = f"{symbol}_RETEST_BUY_{level_name}_{buy_st['break_idx']}"
                            if sig_key not in alerted_this_cycle:
                                signals.append({
                                    "symbol": symbol, "type": "BUY", "entry": "RETEST",
                                    "pivot_name": level_name, "pivot_price": level_price,
                                    "cmp": round(candle["close"], 6),
                                    "ema9": round(ema9_i, 6), "ema26": round(candle["ema26"], 6),
                                    "time": candle_time_str(i),
                                })
                                alerted_this_cycle.add(sig_key)
                        buy_st["phase"], buy_st["break_idx"], buy_st["cross_idx"] = "idle", None, None

        # ── AGGRESSIVE path: crossover-first, break-second, ONE-SHOT per crossover ──
        # Uses candle TIMESTAMP (not DataFrame index) to track which crossover
        # was last used — timestamps are stable across hourly scans even as
        # the DataFrame window slides, unlike row indices which shift each scan.
        if is_latest:
            lookback_start = max(0, i - MAX_WAIT_FOR_CROSS)
            window = df.iloc[lookback_start:i + 1]

            # Most recent bear/bull crossover — get its TIMESTAMP for stable tracking
            bear_rows = window[window["bear_cross"]]
            most_recent_bear_cross_ts = (
                float(df.iloc[bear_rows.index[-1]]["time"])
                if not bear_rows.empty else None
            )

            bull_rows = window[window["bull_cross"]]
            most_recent_bull_cross_ts = (
                float(df.iloc[bull_rows.index[-1]]["time"])
                if not bull_rows.empty else None
            )

            for level_name, level_price in pivot_levels.items():
                prev_close_agg = df.iloc[i - 1]["close"]
                body_size_agg  = abs(candle["close"] - candle["open"])

                if body_size_agg > 0:
                    body_below_agg = (level_price - candle["close"]) / body_size_agg
                    body_above_agg = (candle["close"] - level_price) / body_size_agg
                else:
                    body_below_agg = body_above_agg = 0.0

                clean_break_below = (
                    prev_close_agg  >= level_price and
                    candle["close"] <  level_price and
                    body_below_agg  >= MIN_BODY_BREAK_PCT
                )
                clean_break_above = (
                    prev_close_agg  <= level_price and
                    candle["close"] >  level_price and
                    body_above_agg  >= MIN_BODY_BREAK_PCT
                )

                # SELL: fire only if the crossover timestamp hasn't been used before
                # AND EMA9 is still below EMA26 at trigger (alignment confirmed)
                if (clean_break_below
                        and candle["ema9"] < candle["ema26"]
                        and most_recent_bear_cross_ts is not None
                        and most_recent_bear_cross_ts != state[level_name]["_agg_bear_cross_used"]):
                    sig_key = f"{symbol}_AGG_SELL_{level_name}_{most_recent_bear_cross_ts}"
                    if sig_key not in alerted_this_cycle:
                        signals.append({
                            "symbol": symbol, "type": "SELL", "entry": "AGGRESSIVE",
                            "pivot_name": level_name, "pivot_price": level_price,
                            "cmp": round(candle["close"], 6),
                            "ema9": round(candle["ema9"], 6), "ema26": round(candle["ema26"], 6),
                            "time": candle_time_str(i),
                        })
                        alerted_this_cycle.add(sig_key)
                        state[level_name]["_agg_bear_cross_used"] = most_recent_bear_cross_ts

                # BUY: same one-shot-per-crossover rule using timestamp
                # AND EMA9 must still be above EMA26 at trigger (alignment confirmed)
                if (clean_break_above
                        and candle["ema9"] > candle["ema26"]
                        and most_recent_bull_cross_ts is not None
                        and most_recent_bull_cross_ts != state[level_name]["_agg_bull_cross_used"]):
                    sig_key = f"{symbol}_AGG_BUY_{level_name}_{most_recent_bull_cross_ts}"
                    if sig_key not in alerted_this_cycle:
                        signals.append({
                            "symbol": symbol, "type": "BUY", "entry": "AGGRESSIVE",
                            "pivot_name": level_name, "pivot_price": level_price,
                            "cmp": round(candle["close"], 6),
                            "ema9": round(candle["ema9"], 6), "ema26": round(candle["ema26"], 6),
                            "time": candle_time_str(i),
                        })
                        alerted_this_cycle.add(sig_key)
                        state[level_name]["_agg_bull_cross_used"] = most_recent_bull_cross_ts

    state["_last_idx"] = curr_idx + 1
    return signals




def detect_sniper_signals(symbol, df, pivots):
    """
    SNIPER BUY / SNIPER SELL — Precision entry signal.

    Fully stateless — consolidation is detected by looking back through
    the df directly on every scan. No reliance on in-memory state means
    Railway restarts / missed scans never cause missed signals.

    SNIPER BUY (1H TF + 30min alignment):
      1. Latest candle closes ABOVE pivot AND above EMA9
      2. Looking back from the candle BEFORE trigger — at least 5 of the
         last 30 candles closed within 2% below the pivot (consolidation)
      3. EMA9 at trigger > EMA9 at start of that consolidation window
         AND EMA9 at trigger > EMA9 at midpoint of consolidation (gradual curl)
      4. 30min: EMA9 currently above EMA26

    SNIPER SELL (mirror):
      1. Latest candle closes BELOW pivot AND below EMA9
      2. At least 5 of last 30 candles closed within 2% above pivot
      3. EMA9 curling down during consolidation
      4. 30min: EMA9 currently below EMA26
    """
    if df is None or len(df) < EMA_SLOW + 15:
        return []

    if "time" not in df.columns:
        raise ValueError(f"detect_sniper_signals({symbol}): DataFrame missing 'time' column.")

    signals  = []
    curr_idx = len(df) - 1

    df["ema9"]  = calculate_ema(df["close"], EMA_FAST)
    df["ema26"] = calculate_ema(df["close"], EMA_SLOW)

    def candle_time_str(idx):
        ts = df.iloc[idx]["time"]
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    pivot_levels = {
        "R3": pivots["R3"], "R2": pivots["R2"], "R1": pivots["R1"],
        "P" : pivots["P"],
        "S1": pivots["S1"], "S2": pivots["S2"], "S3": pivots["S3"]
    }

    PROXIMITY_PCT       = 0.035  # candle must close within 3.5% of pivot to count as consolidation
    MIN_BODY_CLOSE_PCT  = 0.40  # at least 40% of candle body must close beyond pivot
    MIN_CONSOL_CANDLES = 5     # minimum consolidation candles required
    MAX_LOOKBACK       = 15    # how far back to scan for consolidation candles
    MAX_BODY_RANGE_PCT = 0.40  # avg candle body must be < 40% of total consol range (no zig-zag)

    def is_tight_consolidation(indices):
        """
        Returns True if consolidation candles are tight/ranging — not zig-zagging.
        Checks that average candle body size is less than 40% of the total
        price range across all consolidation candles.
        A zig-zag market has big bodies relative to its range.
        A tight consolidation has small bodies relative to its range.
        """
        if len(indices) < 2:
            return False
        bodies = []
        closes = []
        for idx in indices:
            c = df.iloc[idx]
            bodies.append(abs(c["close"] - c["open"]))
            closes.append(c["close"])
        avg_body     = sum(bodies) / len(bodies)
        consol_range = max(closes) - min(closes)
        if consol_range == 0:
            return False
        return (avg_body / consol_range) < MAX_BODY_RANGE_PCT

    candle  = df.iloc[curr_idx]
    ema9_c  = candle["ema9"]
    ema26_c = candle["ema26"]

    for level_name, level_price in pivot_levels.items():

        # ══ SNIPER BUY ═══════════════════════════════════════════════════════
        # Trigger: current candle closes above pivot AND above EMA9
        if candle["close"] > level_price and candle["close"] > ema9_c:

            # Gate: 1H EMA9 must be ABOVE EMA26 at trigger — confirms bullish
            # alignment on 1H itself, not just price vs EMA9 in isolation.
            if ema9_c <= ema26_c:
                continue

            # Gate: at least 40% of candle body must close above pivot
            body_size = abs(candle["close"] - candle["open"])
            body_above = candle["close"] - level_price
            if body_size == 0 or (body_above / body_size) < MIN_BODY_CLOSE_PCT:
                continue

            # Gate: EMA9 must be rising into the trigger candle
            # Net rising over last 3 candles + rising in at least one step
            ema9_prev1 = df.iloc[curr_idx - 1]["ema9"]
            ema9_prev2 = df.iloc[curr_idx - 2]["ema9"]
            ema9_rising_into_trigger = (
                ema9_c > ema9_prev2 and  # net rising over 3 candles
                (ema9_c > ema9_prev1 or ema9_prev1 > ema9_prev2)  # rising in at least one step
            )

            if not ema9_rising_into_trigger:
                continue

            # Look back through candles BEFORE the trigger
            # Every candle must close strictly BELOW pivot AND within 5%
            # Any candle outside this zone immediately stops the count — strictly contiguous
            consol_indices = []
            lookback_start = max(curr_idx - MAX_LOOKBACK, EMA_SLOW)

            for j in range(curr_idx - 1, lookback_start - 1, -1):
                c = df.iloc[j]["close"]
                if level_price * (1 - PROXIMITY_PCT) <= c < level_price:
                    consol_indices.append(j)
                else:
                    # Any candle outside zone — stop immediately
                    # Handles: candle above pivot, candle too far below
                    break

            consol_count = len(consol_indices)

            if consol_count >= MIN_CONSOL_CANDLES and is_tight_consolidation(consol_indices):
                # EMA9 slant check — from earliest consol candle to trigger
                earliest_consol_idx = min(consol_indices)
                mid_consol_idx      = (earliest_consol_idx + curr_idx) // 2

                ema9_at_start = df.iloc[earliest_consol_idx]["ema9"]
                ema9_at_mid   = df.iloc[mid_consol_idx]["ema9"]

                # Gradual curl: must be rising overall AND in second half
                # Relaxed: only need ONE of the two to pass if consol is short
                ema9_rising_overall     = ema9_c > ema9_at_start
                ema9_rising_second_half = ema9_c > ema9_at_mid

                if consol_count >= 8:
                    ema9_slanting_up = ema9_rising_overall and ema9_rising_second_half
                else:
                    ema9_slanting_up = ema9_rising_overall

                if ema9_slanting_up:
                    aligned_30m = check_30m_ema_alignment(symbol, "BUY")

                    if aligned_30m:
                        sig_key = f"{symbol}_SNIPER_BUY_{level_name}_{earliest_consol_idx}"
                        if sig_key not in alerted_this_cycle:
                            signals.append({
                                "symbol"        : symbol,
                                "type"          : "BUY",
                                "entry"         : "SNIPER",
                                "pivot_name"    : level_name,
                                "pivot_price"   : level_price,
                                "cmp"           : round(candle["close"], 6),
                                "ema9"          : round(ema9_c, 6),
                                "ema26"         : round(ema26_c, 6),
                                "consol_candles": consol_count,
                                "time"          : candle_time_str(curr_idx),
                            })
                            alerted_this_cycle.add(sig_key)

        # ══ SNIPER SELL ══════════════════════════════════════════════════════
        # Trigger: current candle closes below pivot AND below EMA9
        if candle["close"] < level_price and candle["close"] < ema9_c:

            # Gate: 1H EMA9 must be BELOW EMA26 at trigger — confirms bearish
            # alignment on 1H itself, not just price vs EMA9 in isolation.
            if ema9_c >= ema26_c:
                continue

            # Gate: at least 40% of candle body must close below pivot
            body_size = abs(candle["close"] - candle["open"])
            body_below = level_price - candle["close"]
            if body_size == 0 or (body_below / body_size) < MIN_BODY_CLOSE_PCT:
                continue

            # Gate: EMA9 must be falling into the trigger candle
            ema9_prev1 = df.iloc[curr_idx - 1]["ema9"]
            ema9_prev2 = df.iloc[curr_idx - 2]["ema9"]
            ema9_falling_into_trigger = (
                ema9_c < ema9_prev2 and  # net falling over 3 candles
                (ema9_c < ema9_prev1 or ema9_prev1 < ema9_prev2)  # falling in at least one step
            )

            if not ema9_falling_into_trigger:
                continue

            consol_indices = []
            lookback_start = max(curr_idx - MAX_LOOKBACK, EMA_SLOW)

            for j in range(curr_idx - 1, lookback_start - 1, -1):
                c = df.iloc[j]["close"]
                if level_price < c <= level_price * (1 + PROXIMITY_PCT):
                    consol_indices.append(j)
                else:
                    # Any candle outside zone — stop immediately
                    break

            consol_count = len(consol_indices)

            if consol_count >= MIN_CONSOL_CANDLES and is_tight_consolidation(consol_indices):
                earliest_consol_idx = min(consol_indices)
                mid_consol_idx      = (earliest_consol_idx + curr_idx) // 2

                ema9_at_start = df.iloc[earliest_consol_idx]["ema9"]
                ema9_at_mid   = df.iloc[mid_consol_idx]["ema9"]

                ema9_falling_overall     = ema9_c < ema9_at_start
                ema9_falling_second_half = ema9_c < ema9_at_mid

                if consol_count >= 8:
                    ema9_slanting_down = ema9_falling_overall and ema9_falling_second_half
                else:
                    ema9_slanting_down = ema9_falling_overall

                if ema9_slanting_down:
                    aligned_30m = check_30m_ema_alignment(symbol, "SELL")

                    if aligned_30m:
                        sig_key = f"{symbol}_SNIPER_SELL_{level_name}_{earliest_consol_idx}"
                        if sig_key not in alerted_this_cycle:
                            signals.append({
                                "symbol"        : symbol,
                                "type"          : "SELL",
                                "entry"         : "SNIPER",
                                "pivot_name"    : level_name,
                                "pivot_price"   : level_price,
                                "cmp"           : round(candle["close"], 6),
                                "ema9"          : round(ema9_c, 6),
                                "ema26"         : round(ema26_c, 6),
                                "consol_candles": consol_count,
                                "time"          : candle_time_str(curr_idx),
                            })
                            alerted_this_cycle.add(sig_key)

    return signals


def detect_impulse_signals(symbol, df):
    """
    IMPULSE BUY / IMPULSE SELL — 4th signal type.

    Pattern (BUY side):
      LEG 1  : 3-5 candles with a sharp net upward move.
               Max 1 small red candle allowed within the leg.
               A red candle is "small" if its body is < 50% of the avg
               green candle body in the same leg.
               EMA9 > EMA26 throughout leg 1.

      CONSOL : 4-10 candles of tight consolidation near the TOP of leg 1.
               Price must stay within the upper 60% of leg 1's range
               (not collapsing back down).
               EMA9 > EMA26 maintained throughout.
               Tightness: avg body < 50% of leg 1 avg body (smaller candles).

      TRIGGER: Single candle whose BODY (open-to-close) engulfs the ENTIRE
               consolidation range (body open <= consol low AND body close >= consol high).
               EMA9 > EMA26 at trigger.
               Entry on close of this candle.

    SELL side is the exact mirror.
    """
    if df is None or len(df) < EMA_SLOW + 20:
        return []

    signals  = []
    curr_idx = len(df) - 1

    df["ema9"]  = calculate_ema(df["close"], EMA_FAST)
    df["ema26"] = calculate_ema(df["close"], EMA_SLOW)

    def candle_time_str(idx):
        ts = df.iloc[idx]["time"]
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Constants ────────────────────────────────────────────────────────────
    LEG_MIN       = 3    # minimum candles in leg 1
    LEG_MAX       = 5    # maximum candles in leg 1
    CONSOL_MIN    = 4    # minimum consolidation candles
    CONSOL_MAX    = 10   # maximum consolidation candles
    MAX_OPP_BODY_PCT = 0.50  # opposite candle body must be < 50% of avg leg body

    trigger_candle = df.iloc[curr_idx]
    ema9_t  = trigger_candle["ema9"]
    ema26_t = trigger_candle["ema26"]

    # ══════════════════════════════════════════════════════════════════════════
    # IMPULSE BUY
    # ══════════════════════════════════════════════════════════════════════════
    if ema9_t > ema26_t:  # EMA9 above EMA26 at trigger

        # Scan possible consolidation end points ending just before trigger
        for consol_len in range(CONSOL_MIN, CONSOL_MAX + 1):
            consol_end   = curr_idx - 1          # last consol candle
            consol_start = curr_idx - consol_len  # first consol candle

            if consol_start <= EMA_SLOW + LEG_MAX:
                continue

            consol_df = df.iloc[consol_start:consol_end + 1]

            # Consolidation EMA alignment — EMA9 > EMA26 throughout
            if not (consol_df["ema9"] > consol_df["ema26"]).all():
                continue

            # Consolidation range
            consol_high = consol_df["close"].max()
            consol_low  = consol_df["close"].min()
            consol_range = consol_high - consol_low
            if consol_range == 0:
                continue

            # Consolidation tightness — avg body must be smaller than leg bodies
            consol_avg_body = consol_df.apply(
                lambda r: abs(r["close"] - r["open"]), axis=1
            ).mean()

            # ── Scan possible leg 1 lengths ───────────────────────────────
            for leg_len in range(LEG_MIN, LEG_MAX + 1):
                leg_end   = consol_start - 1
                leg_start = leg_end - leg_len + 1

                if leg_start < EMA_SLOW:
                    continue

                leg_df = df.iloc[leg_start:leg_end + 1]

                # EMA9 > EMA26 throughout leg 1
                if not (leg_df["ema9"] > leg_df["ema26"]).all():
                    continue

                # Identify green and red candles in leg
                green_candles = leg_df[leg_df["close"] > leg_df["open"]]
                red_candles   = leg_df[leg_df["close"] <= leg_df["open"]]

                # Max 1 red candle in leg
                if len(red_candles) > 1:
                    continue

                # Must have at least 2 green candles
                if len(green_candles) < 2:
                    continue

                # Red candle must be small vs avg green body
                avg_green_body = (green_candles["close"] - green_candles["open"]).mean()
                if len(red_candles) == 1:
                    red_body = abs(
                        red_candles.iloc[0]["close"] - red_candles.iloc[0]["open"]
                    )
                    if red_body >= avg_green_body * MAX_OPP_BODY_PCT:
                        continue  # red candle too large — not a small pullback

                # Leg 1 net move must be significant
                leg_net_move = leg_df.iloc[-1]["close"] - leg_df.iloc[0]["open"]
                if leg_net_move <= 0:
                    continue  # net move must be upward

                # Leg 1 move must be larger than consolidation range
                # (ensures the move was sharp relative to the consolidation)
                if leg_net_move <= consol_range:
                    continue

                # Consolidation avg body must be smaller than leg avg body
                # (ensures consolidation is tighter than the impulse leg)
                leg_avg_body = leg_df.apply(
                    lambda r: abs(r["close"] - r["open"]), axis=1
                ).mean()
                if consol_avg_body >= leg_avg_body:
                    continue

                # Price must stay in upper 60% of leg 1 range during consolidation
                leg_high = leg_df["high"].max()
                leg_low  = leg_df["low"].min()
                leg_range = leg_high - leg_low
                if leg_range == 0:
                    continue
                upper_60_threshold = leg_low + leg_range * 0.40
                if consol_low < upper_60_threshold:
                    continue  # consolidation dipped too low into leg 1

                # ── Trigger candle check ──────────────────────────────────
                # Body must engulf entire consolidation range
                trig_body_low  = min(trigger_candle["open"], trigger_candle["close"])
                trig_body_high = max(trigger_candle["open"], trigger_candle["close"])

                body_engulfs = (
                    trig_body_low  <= consol_low and
                    trig_body_high >= consol_high
                )

                if not body_engulfs:
                    continue

                # Must be a green trigger candle
                if trigger_candle["close"] <= trigger_candle["open"]:
                    continue

                sig_key = f"{symbol}_IMPULSE_BUY_{leg_start}_{consol_start}"
                if sig_key not in alerted_this_cycle:
                    signals.append({
                        "symbol"      : symbol,
                        "type"        : "BUY",
                        "entry"       : "IMPULSE",
                        "pivot_name"  : "—",
                        "pivot_price" : 0,
                        "cmp"         : round(trigger_candle["close"], 6),
                        "ema9"        : round(ema9_t, 6),
                        "ema26"       : round(ema26_t, 6),
                        "leg_candles" : leg_len,
                        "consol_candles": consol_len,
                        "consol_high" : round(consol_high, 6),
                        "consol_low"  : round(consol_low, 6),
                        "time"        : candle_time_str(curr_idx),
                    })
                    alerted_this_cycle.add(sig_key)
                break  # found valid setup for this consol_len — no need to try more leg lengths
            else:
                continue
            break  # found valid consol_len — stop scanning

    # ══════════════════════════════════════════════════════════════════════════
    # IMPULSE SELL (exact mirror)
    # ══════════════════════════════════════════════════════════════════════════
    if ema9_t < ema26_t:  # EMA9 below EMA26 at trigger

        for consol_len in range(CONSOL_MIN, CONSOL_MAX + 1):
            consol_end   = curr_idx - 1
            consol_start = curr_idx - consol_len

            if consol_start <= EMA_SLOW + LEG_MAX:
                continue

            consol_df = df.iloc[consol_start:consol_end + 1]

            if not (consol_df["ema9"] < consol_df["ema26"]).all():
                continue

            consol_high = consol_df["close"].max()
            consol_low  = consol_df["close"].min()
            consol_range = consol_high - consol_low
            if consol_range == 0:
                continue

            consol_avg_body = consol_df.apply(
                lambda r: abs(r["close"] - r["open"]), axis=1
            ).mean()

            for leg_len in range(LEG_MIN, LEG_MAX + 1):
                leg_end   = consol_start - 1
                leg_start = leg_end - leg_len + 1

                if leg_start < EMA_SLOW:
                    continue

                leg_df = df.iloc[leg_start:leg_end + 1]

                if not (leg_df["ema9"] < leg_df["ema26"]).all():
                    continue

                red_candles   = leg_df[leg_df["close"] < leg_df["open"]]
                green_candles = leg_df[leg_df["close"] >= leg_df["open"]]

                if len(green_candles) > 1:
                    continue

                if len(red_candles) < 2:
                    continue

                avg_red_body = (red_candles["open"] - red_candles["close"]).mean()
                if len(green_candles) == 1:
                    green_body = abs(
                        green_candles.iloc[0]["close"] - green_candles.iloc[0]["open"]
                    )
                    if green_body >= avg_red_body * MAX_OPP_BODY_PCT:
                        continue

                leg_net_move = leg_df.iloc[0]["open"] - leg_df.iloc[-1]["close"]
                if leg_net_move <= 0:
                    continue

                if leg_net_move <= consol_range:
                    continue

                leg_avg_body = leg_df.apply(
                    lambda r: abs(r["close"] - r["open"]), axis=1
                ).mean()
                if consol_avg_body >= leg_avg_body:
                    continue

                # Price must stay in lower 60% of leg 1 range during consolidation
                leg_high = leg_df["high"].max()
                leg_low  = leg_df["low"].min()
                leg_range = leg_high - leg_low
                if leg_range == 0:
                    continue
                lower_60_threshold = leg_high - leg_range * 0.40
                if consol_high > lower_60_threshold:
                    continue

                # Trigger body must engulf entire consolidation range downward
                trig_body_low  = min(trigger_candle["open"], trigger_candle["close"])
                trig_body_high = max(trigger_candle["open"], trigger_candle["close"])

                body_engulfs = (
                    trig_body_low  <= consol_low and
                    trig_body_high >= consol_high
                )

                if not body_engulfs:
                    continue

                # Must be a red trigger candle
                if trigger_candle["close"] >= trigger_candle["open"]:
                    continue

                sig_key = f"{symbol}_IMPULSE_SELL_{leg_start}_{consol_start}"
                if sig_key not in alerted_this_cycle:
                    signals.append({
                        "symbol"      : symbol,
                        "type"        : "SELL",
                        "entry"       : "IMPULSE",
                        "pivot_name"  : "—",
                        "pivot_price" : 0,
                        "cmp"         : round(trigger_candle["close"], 6),
                        "ema9"        : round(ema9_t, 6),
                        "ema26"       : round(ema26_t, 6),
                        "leg_candles" : leg_len,
                        "consol_candles": consol_len,
                        "consol_high" : round(consol_high, 6),
                        "consol_low"  : round(consol_low, 6),
                        "time"        : candle_time_str(curr_idx),
                    })
                    alerted_this_cycle.add(sig_key)
                break
            else:
                continue
            break

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────────────────────────────────────
def send_whatsapp(message, signal_context=None):
    """
    Sends WhatsApp via Twilio. If the daily message cap (error 63038) is hit,
    logs the full alert to missed_alerts.log instead of crashing, so no
    signal is silently lost — you can review it manually.
    """
    if TWILIO_SID == "your_twilio_sid":
        print("⚠️  Twilio not configured.")
        return
    try:
        client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
        client.messages.create(body=message, from_=TWILIO_FROM, to=TWILIO_TO)
        print("📱 WhatsApp sent!")
    except Exception as e:
        err_str = str(e)
        if "63038" in err_str or "daily messages limit" in err_str.lower():
            print(f"⚠️  Twilio daily cap hit — logging alert instead of sending.")
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "missed_alerts.log")
            with open(log_path, "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"MISSED (Twilio cap) — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
                f.write(message)
                f.write(f"\n{'='*60}\n")
        else:
            print(f"WhatsApp error: {e}")


def send_telegram(message):
    if TELEGRAM_TOKEN == "paste_your_token_here":
        return
    try:
        url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, json=payload, timeout=10)
        print("📱 Telegram sent!")
    except Exception as e:
        print(f"Telegram error: {e}")


def send_alert(signal, df=None):
    emoji     = "🔴" if signal["type"] == "SELL" else "🟢"
    direction = signal["type"]
    entry     = signal["entry"]
    symbol    = signal["symbol"].replace("USDT", "/USDT")

    msg  = f"{emoji} {direction} SIGNAL — {symbol} [{entry}]\n"
    msg += f"━━━━━━━━━━━━━━━━━━\n"
    msg += f"📌 Coin      : {symbol}\n"
    msg += f"💰 CMP       : ${signal['cmp']}\n"
    msg += f"📊 Pivot     : {signal['pivot_name']} (${signal['pivot_price']})\n"
    msg += f"📈 EMA 9     : ${signal['ema9']}\n"
    msg += f"📉 EMA 26    : ${signal['ema26']}\n"
    msg += f"⏰ Time      : {signal['time']} (1H candle)\n"
    msg += f"━━━━━━━━━━━━━━━━━━\n"

    if entry == "RETEST":
        if direction == "SELL":
            msg += f"📋 Setup: Pivot broke ↓ → EMA cross ↓ → Retest EMA9 (High touched, closed below)\n"
        else:
            msg += f"📋 Setup: Pivot broke ↑ → EMA cross ↑ → Retest EMA9 (Low touched, closed above)\n"
    elif entry == "SNIPER":
        consol = signal.get("consol_candles", "?")
        if direction == "SELL":
            msg += f"📋 Setup: {consol} candles consolidated within 3.5% above {signal['pivot_name']} → EMA9 slanted ↓ → Closed below pivot & EMA9 → 30min EMA aligned ↓\n"
        else:
            msg += f"📋 Setup: {consol} candles consolidated within 3.5% below {signal['pivot_name']} → EMA9 slanted ↑ → Closed above pivot & EMA9 → 30min EMA aligned ↑\n"
    elif entry == "IMPULSE":
        leg    = signal.get("leg_candles", "?")
        consol = signal.get("consol_candles", "?")
        c_high = signal.get("consol_high", "?")
        c_low  = signal.get("consol_low", "?")
        if direction == "SELL":
            msg += f"📋 Setup: {leg}-candle sharp drop (Leg 1) → {consol}-candle tight consolidation near bottom (Low: ${c_low} High: ${c_high}) → Body engulfed consolidation ↓\n"
        else:
            msg += f"📋 Setup: {leg}-candle sharp rise (Leg 1) → {consol}-candle tight consolidation near top (Low: ${c_low} High: ${c_high}) → Body engulfed consolidation ↑\n"
    else:
        if direction == "SELL":
            msg += f"📋 Setup: EMA crossed ↓ → Price closed below {signal['pivot_name']}\n"
        else:
            msg += f"📋 Setup: EMA crossed ↑ → Price closed above {signal['pivot_name']}\n"

    # ── ANNOTATIONS (pure context — never affects whether this signal fires) ──
    if df is not None:
        try:
            from pa_analysis import (get_pa_context, get_volume_context,
                                     get_candle_quality, get_4h_alignment)
            pa  = get_pa_context(df, signal["pivot_price"], direction)
            vol = get_volume_context(df)
            cq  = get_candle_quality(df, direction)
            h4  = get_4h_alignment(signal["symbol"], direction)

            trend_emoji = {"UPTREND": "📈", "DOWNTREND": "📉",
                           "RANGE/CONSOLIDATION": "↔️"}.get(pa["trend"], "❓")
            msg += f"━━━━━━━━━━━━━━━━━━\n"
            msg += f"{trend_emoji} Trend     : {pa['trend']} ({pa['trend_reason']})\n"
            if pa["pa_score"] is not None:
                bd = pa["breakdown"]
                msg += (f"📐 PA Score : {pa['pa_score']}/100 "
                        f"(level respected {bd.get('level_respects','?')}x in last 100h, "
                        f"candle {bd.get('body_vs_avg_ratio','?')}x avg, "
                        f"trend {bd.get('trend_alignment','?')})\n")
            msg += f"{vol['emoji']} Volume    : {vol['label']} ({vol['detail']})\n"
            msg += f"{cq['emoji']}  Close Qual: {cq['label']} ({cq['detail']})\n"
            msg += f"{h4['emoji']} 4H Align  : {h4['label']} — {h4['detail']}\n"
        except Exception as e:
            print(f"  (Annotations skipped: {e})")

    msg += f"{emoji} {direction} {entry} CONFIRMED"

    print(f"\n{'='*50}")
    print(msg)
    print('='*50)

    send_whatsapp(msg)
    send_telegram(msg)


# ─────────────────────────────────────────────────────────────────────────────
# WEEKLY PIVOT CACHE
# ─────────────────────────────────────────────────────────────────────────────
def refresh_weekly_pivots(symbols):
    """
    Refresh pivot levels for all symbols.
    Called once at start and every Monday.
    """
    global weekly_pivots
    print(f"\n  📐 Calculating weekly Fibonacci pivots for {len(symbols)} coins...")
    for symbol in symbols:
        pivots = get_pivots_for_symbol(symbol)
        if pivots:
            weekly_pivots[symbol] = pivots
        time.sleep(0.1)  # Rate limit
    print(f"  ✅ Pivots calculated for {len(weekly_pivots)} coins")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCAN CYCLE
# ─────────────────────────────────────────────────────────────────────────────
def run_scan(symbols):
    global alerted_this_cycle

    now = datetime.now(timezone.utc)
    print(f"\n⏰ Scan at {now.strftime('%Y-%m-%d %H:%M UTC')} | {len(symbols)} coins")

    # Refresh pivots every Monday at midnight
    if now.weekday() == 0 and now.hour == 0:
        refresh_weekly_pivots(symbols)

    all_signals = []

    for symbol in symbols:
        try:
            # Get pivots (from cache)
            pivots = weekly_pivots.get(symbol)
            if not pivots:
                continue

            # Fetch 1H candles
            df = get_1h_candles(symbol, limit=100)
            if df is None or len(df) < 30:
                continue

            # Detect signals
            signals = detect_signals(symbol, df, pivots)
            all_signals.extend(signals)

            # Detect SNIPER signals
            sniper_signals = detect_sniper_signals(symbol, df, pivots)
            all_signals.extend(sniper_signals)

            # Detect IMPULSE signals
            impulse_signals = detect_impulse_signals(symbol, df)
            all_signals.extend(impulse_signals)

            # Send individual alerts
            for sig in signals + sniper_signals + impulse_signals:
                send_alert(sig, df)

            time.sleep(0.1)  # Rate limit

        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
            continue

    if not all_signals:
        print(f"  No signals this cycle.")
    else:
        print(f"\n  ✅ {len(all_signals)} signal(s) found this cycle!")

    # Reset alerted set each new week
    if now.weekday() == 0 and now.hour == 0:
        alerted_this_cycle = set()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  CRYPTO PIVOT SCANNER")
    print("  Fibonacci Pivots | EMA 9/26 | 1H TF | Binance")
    print(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("="*60)

    # Get top 50 coins
    symbols = get_top_50_symbols()
    print(f"\n📋 Scanning: {', '.join([s.replace('USDT','') for s in symbols[:10]])}... and {len(symbols)-10} more")

    # Calculate initial weekly pivots
    refresh_weekly_pivots(symbols)

    # Run first scan immediately
    run_scan(symbols)

    # Schedule every hour at :30 minutes (IST) — matches Binance UTC candle close
    # Binance 1H candles close at :00 UTC = :30 IST (since IST = UTC+5:30)
    schedule.every().hour.at(":01").do(run_scan, symbols=symbols)

    print(f"\n✅ Scanner running! Checks every hour at :01 UTC (1 min after Delta 1H candle close)")
    print(f"📱 WhatsApp alerts active")
    print(f"Press Ctrl+C to stop\n")

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
