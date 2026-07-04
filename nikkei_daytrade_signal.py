# -*- coding: utf-8 -*-
"""
Nikkei225 Daytrade Signal Tool
"""

import os
import sys
import json
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone, timedelta


OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "data.json")
HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "signal_history.jsonl")
MAX_HISTORY_LINES = 20000  # 肥大化防止（5分間隔で約2ヶ月分）


def debug_check_intervals():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=2)
    for test_interval in ("1m", "5m", "15m"):
        try:
            d = yf.Ticker("NIY=F").history(start=start, end=end, interval=test_interval)
            last_time = d.index[-1] if d is not None and not d.empty else "no data"
            print(f"DEBUG interval={test_interval} count={len(d) if d is not None else 0} last_time={last_time}")
        except Exception as e:
            print(f"DEBUG interval={test_interval} error={e}")


STALE_THRESHOLD_MINUTES = 60


def fetch_data(period="5d", interval="5m"):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=5)
    candidates = []
    for ticker in ("NIY=F", "^N225"):
        try:
            df = yf.Ticker(ticker).history(
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
            )
        except Exception as e:
            print(f"DEBUG {ticker} fetch failed - {e}")
            df = None
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert("Asia/Tokyo")
            else:
                df.index = df.index.tz_convert("Asia/Tokyo")
            last_time = df.index[-1]
            age_minutes = (datetime.now(timezone.utc).astimezone(last_time.tzinfo) - last_time).total_seconds() / 60
            print(f"DEBUG {ticker} count={len(df)} last_time={last_time} age={age_minutes:.0f}min")
            if age_minutes <= STALE_THRESHOLD_MINUTES:
                print(f"DATA SOURCE: {ticker} (fresh)")
                return df, ticker
            candidates.append((age_minutes, df, ticker))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        age_minutes, df, ticker = candidates[0]
        print(f"WARNING: all sources stale beyond {STALE_THRESHOLD_MINUTES}min, using freshest {ticker} (age {age_minutes:.0f}min)")
        return df, ticker
    raise RuntimeError("Failed to fetch data. Market may be closed or network issue.")


def add_indicators(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["MA5"] = close.rolling(5).mean()
    df["MA25"] = close.rolling(25).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    df["BB_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    return df


def judge_signal(row, prev_row):
    score = 0
    reasons = []

    if prev_row["MA5"] <= prev_row["MA25"] and row["MA5"] > row["MA25"]:
        score += 3
        reasons.append("MA5 crossed above MA25 (Golden Cross)")
    elif prev_row["MA5"] >= prev_row["MA25"] and row["MA5"] < row["MA25"]:
        score -= 3
        reasons.append("MA5 crossed below MA25 (Dead Cross)")

    if row["RSI"] < 30:
        score += 1.5
        reasons.append(f"RSI={row['RSI']:.1f} (oversold)")
    elif row["RSI"] > 70:
        score -= 1.5
        reasons.append(f"RSI={row['RSI']:.1f} (overbought)")

    if prev_row["MACD_hist"] <= 0 and row["MACD_hist"] > 0:
        score += 1.5
        reasons.append("MACD histogram turned positive")
    elif prev_row["MACD_hist"] >= 0 and row["MACD_hist"] < 0:
        score -= 1.5
        reasons.append("MACD histogram turned negative")

    if row["Close"] <= row["BB_lower"]:
        score += 1
        reasons.append("Price <= Bollinger -2sigma (possible rebound)")
    elif row["Close"] >= row["BB_upper"]:
        score -= 1
        reasons.append("Price >= Bollinger +2sigma (possible pullback)")

    if score >= 3:
        signal = "買い"
    elif score <= -3:
        signal = "売り"
    else:
        signal = "様子見"

    return signal, score, reasons


def calc_stop_price(signal, close_price, atr, atr_mult=3.0):
    if atr is None or pd.isna(atr):
        return None
    if signal == "買い":
        return round(float(close_price - atr * atr_mult), 2)
    elif signal == "売り":
        return round(float(close_price + atr * atr_mult), 2)
    return None


def send_slack_notification(webhook_url, signal, score, reasons, latest, timestamp, stop_price):
    emoji = "green" if signal == "買い" else "red"
    reasons_text = "\n".join(f"- {r}" for r in reasons)
    stop_text = f"\nStop loss (ATRx3): {stop_price:,.2f}" if stop_price else ""
    text = (
        f"[{emoji}] Nikkei225 Signal: {signal} (score: {score:+.1f})\n"
        f"Time: {timestamp}\n"
        f"Close: {latest['Close']:.2f}\n"
        f"MA5: {latest['MA5']:.2f} / MA25: {latest['MA25']:.2f}\n"
        f"RSI: {latest['RSI']:.1f}"
        f"{stop_text}\n"
        f"\nReasons:\n{reasons_text}\n"
    )
    resp = requests.post(webhook_url, json={"text": text}, timeout=10)
    resp.raise_for_status()


def build_json(df, signal, score, reasons, timestamp, ticker, stop_price, timeframes=None, composite=None):
    recent = df.tail(100).copy()
    candles = []
    for ts, row in recent.iterrows():
        candles.append({
            "time": ts.strftime("%Y-%m-%d %H:%M"),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "ma5": None if pd.isna(row["MA5"]) else round(float(row["MA5"]), 2),
            "ma25": None if pd.isna(row["MA25"]) else round(float(row["MA25"]), 2),
            "rsi": None if pd.isna(row["RSI"]) else round(float(row["RSI"]), 1),
            "macd": None if pd.isna(row["MACD"]) else round(float(row["MACD"]), 2),
            "macd_signal": None if pd.isna(row["MACD_signal"]) else round(float(row["MACD_signal"]), 2),
            "macd_hist": None if pd.isna(row["MACD_hist"]) else round(float(row["MACD_hist"]), 2),
            "bb_upper": None if pd.isna(row["BB_upper"]) else round(float(row["BB_upper"]), 2),
            "bb_lower": None if pd.isna(row["BB_lower"]) else round(float(row["BB_lower"]), 2),
        })

    latest = df.iloc[-1]

    data = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "latest_time": timestamp,
        "latest_close": round(float(latest["Close"]), 2),
        "signal": signal,
        "score": score,
        "reasons": reasons,
        "stop_price": stop_price,
        "candles": candles,
        "data_source": ticker,
        "timeframes": timeframes or {},
        "composite": composite,
    }
    return data


def fetch_timeframe_data(interval, days, tickers=("NIY=F", "^N225")):
    """指定した時間軸のデータを取得する(マルチタイムフレーム判定用)"""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
            )
        except Exception as e:
            print(f"DEBUG multi-tf {ticker} {interval} fetch failed - {e}")
            df = None
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert("Asia/Tokyo")
            else:
                df.index = df.index.tz_convert("Asia/Tokyo")
            return df, ticker
    return None, None


def get_timeframe_signal(label, interval, days):
    """指定した時間軸の最新シグナルを算出する"""
    try:
        df, ticker = fetch_timeframe_data(interval, days)
        if df is None:
            return {"label": label, "signal": None, "score": None, "error": "data unavailable"}
        df = add_indicators(df)
        df = df.dropna()
        if len(df) < 2:
            return {"label": label, "signal": None, "score": None, "error": "not enough data"}
        latest_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        signal, score, _reasons = judge_signal(latest_row, prev_row)
        return {
            "label": label,
            "signal": signal,
            "score": score,
            "close": round(float(latest_row["Close"]), 2),
            "time": df.index[-1].strftime("%Y-%m-%d %H:%M"),
            "data_source": ticker,
        }
    except Exception as e:
        print(f"DEBUG multi-tf {label} error: {e}")
        return {"label": label, "signal": None, "score": None, "error": str(e)}


def build_multi_timeframe():
    """5分足・15分足・1時間足の3つを取得して辞書にまとめる"""
    return {
        "15m": get_timeframe_signal("15分足", "15m", days=10),
        "1h": get_timeframe_signal("1時間足", "60m", days=30),
    }


def composite_judgement(main_signal, timeframes):
    """メイン(5分足)＋他時間軸を合わせた総合判定"""
    signals = [main_signal] + [tf["signal"] for tf in timeframes.values() if tf.get("signal")]
    buy_count = signals.count("買い")
    sell_count = signals.count("売り")
    total = len(signals)

    if total == 0:
        return "判定不可"
    if buy_count == total and total > 1:
        return "全時間軸一致(買い)"
    if sell_count == total and total > 1:
        return "全時間軸一致(売り)"
    if buy_count > sell_count:
        return "買い優勢"
    if sell_count > buy_count:
        return "売り優勢"
    return "方向感なし"


def append_history(latest, signal, score, timestamp, ticker):
    """毎回の実行結果を1行ずつ追記していく（バックテスト用の時系列データ）"""
    record = {
        "time": timestamp,
        "close": round(float(latest["Close"]), 2),
        "signal": signal,
        "score": score,
        "atr": None if pd.isna(latest.get("ATR")) else round(float(latest["ATR"]), 2),
    }

    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "rb") as f:
            try:
                f.seek(-200, os.SEEK_END)
            except OSError:
                f.seek(0)
            last_line = f.readlines()[-1].decode("utf-8", errors="ignore").strip()
        if last_line:
            try:
                last_record = json.loads(last_line)
                if last_record.get("time") == timestamp:
                    print(f"history: {timestamp} already logged, skip")
                    return
            except json.JSONDecodeError:
                pass

    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) > MAX_HISTORY_LINES:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            f.writelines(lines[-MAX_HISTORY_LINES:])

    print(f"history appended: {timestamp} ({len(lines)} lines)")


def main():
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")

    debug_check_intervals()

    df, ticker = fetch_data(period="5d", interval="5m")
    df = add_indicators(df)
    df = df.dropna()

    if len(df) < 2:
        print("Not enough data.")
        sys.exit(0)

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    timestamp = df.index[-1].strftime("%Y-%m-%d %H:%M")

    signal, score, reasons = judge_signal(latest, prev)
    stop_price = calc_stop_price(signal, latest["Close"], latest.get("ATR"))

    print(f"[{timestamp}] signal: {signal} (score: {score:+.1f})")
    for r in reasons:
        print(f"  - {r}")
    if stop_price:
        print(f"  stop price (ATRx3): {stop_price:,.2f}")

    timeframes = build_multi_timeframe()
    composite = composite_judgement(signal, timeframes)
    print(f"  composite (5m+15m+1h): {composite}")
    for tf_key, tf_val in timeframes.items():
        print(f"  [{tf_val.get('label', tf_key)}] signal={tf_val.get('signal')} score={tf_val.get('score')}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    data = build_json(df, signal, score, reasons, timestamp, ticker, stop_price, timeframes, composite)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"data written: {OUTPUT_PATH}")

    append_history(latest, signal, score, timestamp, ticker)

    if signal == "様子見":
        print("hold - skipping notification")
        return

    if not webhook_url:
        print("warning: SLACK_WEBHOOK_URL not set, skipping notification")
        return

    send_slack_notification(webhook_url, signal, score, reasons, latest, timestamp, stop_price)
    print("Slack notification sent.")


if __name__ == "__main__":
    main()
