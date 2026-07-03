# -*- coding: utf-8 -*-
"""
マイクロ日経225 デイトレード エントリー判断ツール (ダッシュボード対応版)
========================================================================

実行のたびに、最新のローソク足・指標・シグナルを docs/data.json に書き出します。
これを docs/index.html (GitHub Pages) が読み込んでチャート表示します。
あわせて、買い/売りシグナルが出た場合はSlackに通知します。

【必要な環境変数】
  SLACK_WEBHOOK_URL : SlackのIncoming Webhook URL (任意。無くても動作する)

【必要ライブラリ】
  pip install yfinance pandas numpy requests --break-system-packages
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


def debug_check_intervals():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=2)
    for test_interval in ("1m", "5m", "15m"):
        try:
            d = yf.Ticker("NIY=F").history(start=start, end=end, interval=test_interval)
            last_time = d.index[-1] if d is not None and not d.empty else "データなし"
            print(f"デバッグ確認: interval={test_interval} 件数={len(d) if d is not None else 0} 最終時刻={last_time}")
        except Exception as e:
            print(f"デバッグ確認: interval={test_interval} エラー={e}")


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
            print(f"デバッグ: {ticker} 取得失敗 - {e}")
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
            print(f"デバッグ: {ticker} 取得件数={len(df)}, 最終時刻={last_time}, 経過={age_minutes:.0f}分")
            if age_minutes <= STALE_THRESHOLD_MINUTES:
                print(f"データ取得元: {ticker} (鮮度良好)")
                return df, ticker
            candidates.append((age_minutes, df, ticker))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        age_minutes, df, ticker = candidates[0]
        print(f"警告: 全銘柄のデータが{STALE_THRESHOLD_MINUTES}分以上古いため、最も新しい{ticker}(経過{age_minutes:.0f}分)を使用します。")
        return df, ticker
    raise RuntimeError("データ取得に失敗しました。市場が開いていない、もしくはネットワーク接続を確認してください。")


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
        reasons.append("MA5がMA25を上抜け(ゴールデンクロス)")
    elif prev_row["MA5"] >= prev_row["MA25"] and row["MA5"] < row["MA25"]:
        score -= 3
        reasons.append("MA5がMA25を下抜け(デッドクロス)")

    if row["RSI"] < 30:
        score += 1.5
        reasons.append(f"RSI={row['RSI']:.1f}(売られすぎ圏)")
    elif row["RSI"] > 70:
        score -= 1.5
        reasons.append(f"RSI={row['RSI']:.1f}(買われすぎ圏)")

    if prev_row["MACD_hist"] <= 0 and row["MACD_hist"] > 0:
        score += 1.5
        reasons.append("MACDヒストグラムがプラス転換")
    elif prev_row["MACD_hist"] >= 0 and row["MACD_hist"] < 0:
        score -= 1.5
        reasons.append("MACDヒストグラムがマイナス転換")

    if row["Close"] <= row["BB_lower"]:
        score += 1
        reasons.append("price <= ボリンジャーバンド-2σ(反発の可能性)")
    elif row["Close"] >= row["BB_upper"]:
        score -= 1
        reasons.append("price >= ボリンジャーバンド+2σ(反落の可能性)")

    if score >= 3:
        signal = "買い"
    elif score <= -3:
        signal = "売り"
    else:
        signal = "様子見"

    return signal, score, reasons


def calc_stop_price(signal, close_price, atr, atr_mult=3.0):
    """ATRベースの推奨損切りラインを計算する(参考値、判定には使わない)"""
    if atr is None or pd.isna(atr):
        return None
    if signal == "買い":
        return round(float(close_price - atr * atr_mult), 2)
    elif signal == "売り":
        return round(float(close_price + atr * atr_mult), 2)
    return None


def send_slack_notification(webhook_url, signal, score, reasons, latest, timestamp, stop_price):
    emoji = "🟢" if signal == "買い" else "🔴"
    reasons_text = "\n".join(f"・{r}" for r in reasons)
    stop_text = f"\n参考損切りライン(ATR×3): {stop_price:,.2f}" if stop_price else ""
    text = (
        f"{emoji} *マイクロ日経225 シグナル: {signal}* (スコア: {score:+.1f})\n"
        f"時刻: {timestamp}\n"
        f"日経平均: {latest['Close']:.2f}\n"
        f"MA5: {latest['MA5']:.2f} / MA25: {latest['MA25']:.2f}\n"
        f"RSI: {latest['RSI']:.1f}"
        f"{stop_text}\n"
        f"\n根拠:\n{reasons_text}\n"
        f"\n※テクニカル指標による参考情報です。投資判断は自己責任で行ってください。"
    )
    resp = requests.post(webhook_url, json={"text": text}, timeout=10)
    resp.raise_for_status()


def build_json(df, signal, score, reasons, timestamp, ticker, stop_price):
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
    }
    return data


def main():
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")

    debug_check_intervals()

    df, ticker = fetch_data(period="5d", interval="5m")
    df = add_indicators(df)
    df = df.dropna()

    if len(df) < 2:
        print("データが不足しています。")
        sys.exit(0)

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    timestamp = df.index[-1].strftime("%Y-%m-%d %H:%M")

    signal, score, reasons = judge_signal(latest, prev)
    stop_price = calc_stop_price(signal, latest["Close"], latest.get("ATR"))

    print(f"[{timestamp}] 判定: {signal} (スコア: {score:+.1f})")
    for r in reasons:
        print(f"  - {r}")
    if stop_price:
        print(f"  参考損切りライン(ATR×3): {stop_price:,.2f}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    data = build_json(df, signal, score, reasons, timestamp, ticker, stop_price)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"データを書き出しました: {OUTPUT_PATH}")

    if signal == "様子見":
        print("様子見のため通知はスキップします。")
        return

    if not webhook_url:
        print("警告: SLACK_WEBHOOK_URLが設定されていないため通知をスキップします。")
        return

    send_slack_notification(webhook_url, signal, score, reasons, latest, timestamp, stop_price)
    print("Slackに通知しました。")


if __name__ == "__main__":
    main()
