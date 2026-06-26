# =============================================================================
# pa_analysis.py — Price Action Context (ANNOTATION ONLY)
#
# IMPORTANT: This module does NOT affect signal generation in any way. It is
# called AFTER detect_signals() has already decided to fire a signal, purely
# to add informational context to the alert message. It never gates, filters,
# or modifies whether a signal fires. If this module errors, the signal still
# fires normally — just without the extra context line.
#
# Provides two pieces of context per signal:
#   1. TREND — based on swing high/low structure (higher highs/lows = uptrend,
#      lower highs/lows = downtrend, mixed = range/consolidation)
#   2. PA SCORE — a 0-100 confidence score combining:
#        - How many times the broken pivot level was previously
#          respected (touched and bounced) before this break
#        - Strength of the breakout candle (body size vs recent average)
#        - Whether the break aligns with or goes against the swing trend
# =============================================================================

import pandas as pd
import numpy as np


def find_swing_points(df, lookback=3):
    """
    Identify swing highs and swing lows using a simple fractal method:
    a candle's high is a swing high if it's higher than `lookback` candles
    on each side; same logic (inverted) for swing lows.

    Returns two lists of (index, price) tuples.
    """
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    swing_highs = []
    swing_lows  = []

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback:i + lookback + 1]
        window_l = lows[i - lookback:i + lookback + 1]

        if highs[i] == window_h.max() and highs[i] > 0:
            swing_highs.append((i, highs[i]))
        if lows[i] == window_l.min() and lows[i] > 0:
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def determine_trend(df, lookback=3, recent_swings=3):
    """
    Classify trend based on the last `recent_swings` swing highs and lows.

    Returns one of: "UPTREND", "DOWNTREND", "RANGE/CONSOLIDATION"
    plus a short reason string.
    """
    try:
        swing_highs, swing_lows = find_swing_points(df, lookback=lookback)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "RANGE/CONSOLIDATION", "Not enough swing points yet to confirm structure"

        recent_highs = [p for _, p in swing_highs[-recent_swings:]]
        recent_lows  = [p for _, p in swing_lows[-recent_swings:]]

        highs_rising = all(recent_highs[i] < recent_highs[i + 1] for i in range(len(recent_highs) - 1))
        lows_rising  = all(recent_lows[i]  < recent_lows[i + 1]  for i in range(len(recent_lows) - 1))

        highs_falling = all(recent_highs[i] > recent_highs[i + 1] for i in range(len(recent_highs) - 1))
        lows_falling  = all(recent_lows[i]  > recent_lows[i + 1]  for i in range(len(recent_lows) - 1))

        if highs_rising and lows_rising:
            return "UPTREND", f"Higher highs ({len(recent_highs)}) and higher lows ({len(recent_lows)}) confirmed"
        elif highs_falling and lows_falling:
            return "DOWNTREND", f"Lower highs ({len(recent_highs)}) and lower lows ({len(recent_lows)}) confirmed"
        else:
            return "RANGE/CONSOLIDATION", "Mixed swing structure — no clear higher-highs/lows or lower-highs/lows pattern"

    except Exception as e:
        return "UNKNOWN", f"Trend analysis error: {e}"


def count_level_respects(df, level_price, lookback_candles=100, touch_buffer_pct=0.0015):
    """
    Count how many times price approached this pivot level and reversed
    (i.e. respected it as support/resistance) in the recent history,
    BEFORE the most recent candle (which is the breaking candle itself).

    A "respect" = price's high or low came within touch_buffer_pct of the
    level, and the candle closed back on the same side it approached from
    (didn't close through the level).
    """
    try:
        recent = df.iloc[-(lookback_candles + 1):-1]  # exclude current/breaking candle
        if len(recent) < 5:
            return 0

        respects = 0
        for _, c in recent.iterrows():
            near_level = (
                abs(c["high"] - level_price) <= level_price * touch_buffer_pct or
                abs(c["low"]  - level_price) <= level_price * touch_buffer_pct
            )
            if near_level:
                # Respected if close stayed on one side (didn't break through)
                closed_above = c["close"] > level_price
                closed_below = c["close"] < level_price
                if closed_above or closed_below:
                    respects += 1

        return respects

    except Exception:
        return 0


def calculate_pa_score(df, level_price, signal_type, trend):
    """
    Composite 0-100 score combining:
      - Level respect count (0-40 points): more prior respects = stronger level = higher score
      - Breakout candle strength (0-30 points): bigger body vs recent average = higher score
      - Trend alignment (0-30 points): signal direction matching swing trend = higher score

    Returns (score, breakdown_dict)
    """
    breakdown = {}

    try:
        # ── Component 1: Level respect count (0-40) ──────────────────────────
        respects = count_level_respects(df, level_price)
        respect_score = min(respects * 10, 40)  # cap at 40, 4+ respects = max
        breakdown["level_respects"] = respects
        breakdown["respect_score"] = respect_score

        # ── Component 2: Breakout candle strength (0-30) ──────────────────────
        current = df.iloc[-1]
        body_size = abs(current["close"] - current["open"])

        recent_bodies = (df.iloc[-21:-1]["close"] - df.iloc[-21:-1]["open"]).abs()
        avg_body = recent_bodies.mean() if len(recent_bodies) > 0 else body_size

        if avg_body > 0:
            body_ratio = body_size / avg_body
            candle_score = min(body_ratio * 15, 30)  # 2x avg body = max score
        else:
            candle_score = 0
        breakdown["body_vs_avg_ratio"] = round(body_size / avg_body, 2) if avg_body > 0 else None
        breakdown["candle_score"] = round(candle_score, 1)

        # ── Component 3: Trend alignment (0-30) ───────────────────────────────
        if signal_type == "BUY" and trend == "UPTREND":
            trend_score = 30
        elif signal_type == "SELL" and trend == "DOWNTREND":
            trend_score = 30
        elif trend == "RANGE/CONSOLIDATION":
            trend_score = 15  # neutral — could go either way
        else:
            # Counter-trend signal (e.g. SELL during UPTREND, like the JTO example)
            trend_score = 5
        breakdown["trend_score"] = trend_score
        breakdown["trend_alignment"] = (
            "aligned" if trend_score == 30 else
            "neutral/ranging" if trend_score == 15 else
            "counter-trend"
        )

        total_score = round(respect_score + candle_score + trend_score)
        return min(total_score, 100), breakdown

    except Exception as e:
        breakdown["error"] = str(e)
        return None, breakdown


def get_pa_context(df, level_price, signal_type):
    """
    Main entry point. Call this AFTER a signal has already been generated.
    Returns a dict with trend, reason, score, and breakdown — or a safe
    fallback dict if anything fails, so the caller can always proceed.
    """
    fallback = {
        "trend": "UNKNOWN",
        "trend_reason": "Could not analyze",
        "pa_score": None,
        "breakdown": {}
    }

    try:
        trend, reason = determine_trend(df)
        score, breakdown = calculate_pa_score(df, level_price, signal_type, trend)
        return {
            "trend": trend,
            "trend_reason": reason,
            "pa_score": score,
            "breakdown": breakdown
        }
    except Exception as e:
        fallback["trend_reason"] = f"PA analysis failed: {e}"
        return fallback


# =============================================================================
# ANNOTATION 1 — VOLUME CONFIRMATION
# Is the breakout candle's volume significantly above recent average?
# High volume on a pivot break = real participation.
# Low volume = thin move, more likely to reverse.
# =============================================================================

def get_volume_context(df, lookback=20):
    """
    Compares the current (signal) candle's volume against the recent average.
    Returns a dict with ratio, label, and emoji for the alert message.
    Never raises — returns a safe fallback on any error.
    """
    try:
        if df is None or len(df) < lookback + 1:
            return {"label": "UNKNOWN", "ratio": None, "emoji": "❓",
                    "detail": "Not enough candles"}

        current_vol = df.iloc[-1]["volume"]
        avg_vol     = df.iloc[-(lookback + 1):-1]["volume"].mean()

        if avg_vol == 0:
            return {"label": "UNKNOWN", "ratio": None, "emoji": "❓",
                    "detail": "Avg volume is zero"}

        ratio = round(current_vol / avg_vol, 2)

        if ratio >= 3.0:
            label, emoji = "VERY HIGH", "🔥"
        elif ratio >= 2.0:
            label, emoji = "HIGH", "✅"
        elif ratio >= 1.2:
            label, emoji = "ABOVE AVG", "📊"
        elif ratio >= 0.8:
            label, emoji = "AVERAGE", "➡️"
        else:
            label, emoji = "LOW", "⚠️"

        return {
            "label" : label,
            "ratio" : ratio,
            "emoji" : emoji,
            "detail": f"{ratio}x the {lookback}-candle average"
        }

    except Exception as e:
        return {"label": "ERROR", "ratio": None, "emoji": "❓",
                "detail": str(e)}


# =============================================================================
# ANNOTATION 2 — CANDLE CLOSE QUALITY
# Did the signal candle close with conviction?
#   BUY signal: candle should close near its HIGH (bullish close)
#   SELL signal: candle should close near its LOW (bearish close)
# A candle that breaks a level but closes in the middle of its range
# shows indecision — less reliable.
# =============================================================================

def get_candle_quality(df, signal_type):
    """
    Measures where the candle closed within its high-low range.
    Returns a dict with score (0-100), label, and emoji.
    """
    try:
        if df is None or len(df) < 1:
            return {"label": "UNKNOWN", "score": None, "emoji": "❓",
                    "detail": "No candle data"}

        c     = df.iloc[-1]
        high  = c["high"]
        low   = c["low"]
        close = c["close"]
        rng   = high - low

        if rng == 0:
            return {"label": "DOJI", "score": 50, "emoji": "➡️",
                    "detail": "No range — doji candle"}

        # Position of close within the candle's range (0 = at low, 1 = at high)
        close_position = (close - low) / rng

        if signal_type == "BUY":
            # For BUY: higher close position = better (closed near high)
            score = round(close_position * 100)
            if score >= 75:
                label, emoji = "STRONG CLOSE", "✅"
            elif score >= 50:
                label, emoji = "MODERATE CLOSE", "➡️"
            else:
                label, emoji = "WEAK CLOSE", "⚠️"
            detail = f"Closed in top {100-score}% of range" if score >= 50 else f"Closed in bottom {score}% of range"
        else:
            # For SELL: lower close position = better (closed near low)
            score = round((1 - close_position) * 100)
            if score >= 75:
                label, emoji = "STRONG CLOSE", "✅"
            elif score >= 50:
                label, emoji = "MODERATE CLOSE", "➡️"
            else:
                label, emoji = "WEAK CLOSE", "⚠️"
            detail = f"Closed in bottom {100-score}% of range" if score >= 50 else f"Closed in top {score}% of range"

        return {
            "label" : label,
            "score" : score,
            "emoji" : emoji,
            "detail": detail
        }

    except Exception as e:
        return {"label": "ERROR", "score": None, "emoji": "❓",
                "detail": str(e)}


# =============================================================================
# ANNOTATION 3 — HIGHER TIMEFRAME (4H) PIVOT ALIGNMENT
# Does the 4H pivot level context agree with the signal direction?
# A SELL signal with the 4H pivot structure also bearish = stronger.
# A BUY signal going against the 4H structure = be cautious.
# Fetches 4H candles from Delta and calculates Fibonacci pivots on weekly 4H OHLC.
# =============================================================================

def get_4h_alignment(symbol, signal_type, delta_base_url="https://api.india.delta.exchange"):
    """
    Fetches the last 50 4H candles from Delta and checks:
    1. Where is price relative to the 4H pivot (P)?
       - Price above 4H P + BUY signal = aligned (bullish on both TFs)
       - Price below 4H P + SELL signal = aligned (bearish on both TFs)
       - Opposite = counter-HTF
    2. What direction is the 4H EMA9 vs EMA26?
    Returns a dict with alignment label and emoji.
    """
    try:
        import requests
        from datetime import datetime, timedelta, timezone

        end_ts   = int(datetime.now(timezone.utc).timestamp())
        start_ts = end_ts - 50 * 4 * 3600  # last 50 4H candles

        url    = f"{delta_base_url}/v2/history/candles"
        params = {"symbol": symbol, "resolution": "4h",
                  "start": start_ts, "end": end_ts}
        resp   = requests.get(url, params=params, timeout=8)
        data   = resp.json()

        if not data.get("success") or not data.get("result"):
            return {"label": "UNAVAILABLE", "emoji": "❓",
                    "detail": "Could not fetch 4H data"}

        candles_4h = data["result"]
        if not isinstance(candles_4h, list) or len(candles_4h) < 10:
            return {"label": "UNAVAILABLE", "emoji": "❓",
                    "detail": f"Only {len(candles_4h)} 4H candles returned"}

        df4 = pd.DataFrame(candles_4h).astype({
            "open": float, "high": float, "low": float,
            "close": float, "volume": float
        })
        df4 = df4.sort_values("time").reset_index(drop=True)

        # Drop forming candle
        now_ts = int(datetime.now(timezone.utc).timestamp())
        now_4h_ts = int(now_ts // (4 * 3600) * (4 * 3600))
        if df4.iloc[-1]["time"] >= now_4h_ts:
            df4 = df4.iloc[:-1].reset_index(drop=True)

        if len(df4) < 5:
            return {"label": "UNAVAILABLE", "emoji": "❓",
                    "detail": "Not enough closed 4H candles"}

        # Simple 4H pivot: use last completed 4H session high/low/close
        # (we use last 6 4H candles = 1 day as the "session")
        session = df4.iloc[-7:-1] if len(df4) >= 7 else df4.iloc[:-1]
        h4_high  = session["high"].max()
        h4_low   = session["low"].min()
        h4_close = session.iloc[-1]["close"]
        h4_pivot = (h4_high + h4_low + h4_close) / 3

        current_price = df4.iloc[-1]["close"]

        # 4H EMA direction
        df4["ema9"]  = df4["close"].ewm(span=9,  adjust=False).mean()
        df4["ema26"] = df4["close"].ewm(span=26, adjust=False).mean()
        h4_ema9  = df4.iloc[-1]["ema9"]
        h4_ema26 = df4.iloc[-1]["ema26"]
        h4_bullish = h4_ema9 > h4_ema26
        h4_bearish = h4_ema9 < h4_ema26

        price_above_pivot = current_price > h4_pivot

        if signal_type == "BUY" and price_above_pivot and h4_bullish:
            label, emoji = "ALIGNED ✅", "📈"
            detail = f"4H price above pivot ({round(h4_pivot,4)}), 4H EMA bullish"
        elif signal_type == "SELL" and not price_above_pivot and h4_bearish:
            label, emoji = "ALIGNED ✅", "📉"
            detail = f"4H price below pivot ({round(h4_pivot,4)}), 4H EMA bearish"
        elif signal_type == "BUY" and not price_above_pivot:
            label, emoji = "COUNTER-HTF ⚠️", "⚠️"
            detail = f"4H price below pivot ({round(h4_pivot,4)}) — buying against 4H structure"
        elif signal_type == "SELL" and price_above_pivot:
            label, emoji = "COUNTER-HTF ⚠️", "⚠️"
            detail = f"4H price above pivot ({round(h4_pivot,4)}) — selling against 4H structure"
        else:
            label, emoji = "NEUTRAL ➡️", "➡️"
            detail = f"4H price {'above' if price_above_pivot else 'below'} pivot ({round(h4_pivot,4)}), EMA mixed"

        return {
            "label" : label,
            "emoji" : emoji,
            "detail": detail,
            "h4_pivot"   : round(h4_pivot, 6),
            "h4_ema_bias": "bullish" if h4_bullish else "bearish" if h4_bearish else "flat"
        }

    except Exception as e:
        return {"label": "ERROR", "emoji": "❓", "detail": str(e)}
