# =============================================================================
# crypto_pivot.py — Fibonacci Pivot Points
# Now sourced from DELTA EXCHANGE INDIA (api.india.delta.exchange)
# — matches the actual exchange the user trades on, not Binance.
#
# Verified against live API response (2026-06-19):
#   GET /v2/history/candles?symbol=JTOUSD&resolution=1w&start=...&end=...
#   Returns: {"success": true, "result": [{"time":..,"open":..,"high":..,
#             "low":..,"close":..,"volume":..}, ...]}
#   Candles returned newest-first.
#
# Fibonacci formulas (unchanged, standard):
#   P  = (High + Low + Close) / 3
#   R1 = P + 0.382 * (High - Low)
#   R2 = P + 0.618 * (High - Low)
#   R3 = P + 1.000 * (High - Low)
#   S1 = P - 0.382 * (High - Low)
#   S2 = P - 0.618 * (High - Low)
#   S3 = P - 1.000 * (High - Low)
#
# NOTE: pivot values calculated here were checked against a TradingView
# chart for JTOUSD and were close but not byte-identical (~0.3% gap on R3).
# Root cause not fully isolated — likely a minor weekly-boundary or
# candle timestamp-convention difference. Worth re-checking periodically
# against the chart rather than assuming permanent exact match.
# =============================================================================

import requests
from datetime import datetime, timedelta, timezone

DELTA_BASE_URL = "https://api.india.delta.exchange"


def get_previous_week_ohlc(symbol):
    """
    Fetch previous completed week's OHLC from Delta Exchange India.
    Uses weekly candles, takes the most recently CLOSED week (not the
    current forming week).
    """
    try:
        end_ts   = int(datetime.now(timezone.utc).timestamp())
        start_ts = int((datetime.now(timezone.utc) - timedelta(weeks=4)).timestamp())

        url    = f"{DELTA_BASE_URL}/v2/history/candles"
        params = {
            "symbol"    : symbol,
            "resolution": "1w",
            "start"     : start_ts,
            "end"       : end_ts
        }
        response = requests.get(url, params=params, timeout=10)
        data     = response.json()

        if not data.get("success"):
            print(f"  ⚠️  Delta API returned success=False for {symbol}: {data}")
            return None

        result = data.get("result")
        if not result:
            print(f"  ⚠️  Delta API returned empty result for {symbol}: {data}")
            return None

        if not isinstance(result, list):
            print(f"  ⚠️  Delta API returned unexpected result type for {symbol}: {type(result)} -> {result}")
            return None

        candles = result  # newest first, per confirmed test

        now = datetime.now(timezone.utc)
        this_monday = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        this_monday_ts = int(this_monday.timestamp())

        completed_weeks = [c for c in candles if c.get("time", 0) < this_monday_ts]
        if not completed_weeks:
            print(f"  ⚠️  No completed-week candle found for {symbol} (got {len(candles)} candles, "
                  f"this_monday_ts={this_monday_ts})")
            return None

        prev_week = completed_weeks[0]

        return {
            "open" : float(prev_week["open"]),
            "high" : float(prev_week["high"]),
            "low"  : float(prev_week["low"]),
            "close": float(prev_week["close"])
        }

    except Exception as e:
        print(f"  Error fetching weekly OHLC for {symbol} from Delta: {type(e).__name__}: {e}")
        return None


def calculate_fibonacci_pivots(ohlc):
    """Calculate Fibonacci Pivot Points from weekly OHLC."""
    if not ohlc:
        return None

    H = ohlc["high"]
    L = ohlc["low"]
    C = ohlc["close"]

    P     = (H + L + C) / 3
    Range = H - L

    pivots = {
        "P" : round(P, 8),
        "R1": round(P + 0.382 * Range, 8),
        "R2": round(P + 0.618 * Range, 8),
        "R3": round(P + 1.000 * Range, 8),
        "S1": round(P - 0.382 * Range, 8),
        "S2": round(P - 0.618 * Range, 8),
        "S3": round(P - 1.000 * Range, 8),
    }

    return pivots


def get_pivots_for_symbol(symbol):
    """
    Main function: fetch weekly OHLC from Delta and calculate Fibonacci pivots.
    Pivots are static for the entire week.
    """
    ohlc   = get_previous_week_ohlc(symbol)
    pivots = calculate_fibonacci_pivots(ohlc)
    return pivots


if __name__ == "__main__":
    symbol = "JTOUSD"
    pivots = get_pivots_for_symbol(symbol)
    if pivots:
        print(f"\nFibonacci Pivots for {symbol} (Weekly, Delta Exchange India):")
        for k in ["R3", "R2", "R1", "P", "S1", "S2", "S3"]:
            print(f"  {k}: {pivots[k]}")
    else:
        print(f"Failed to fetch pivots for {symbol}")
