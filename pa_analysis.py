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
