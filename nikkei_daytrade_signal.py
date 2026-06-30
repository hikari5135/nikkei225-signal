# -*- coding: utf-8 -*-
"""
マイクロ日経225 デイトレード エントリー判断ツール (Slack通知版)
==============================================================

GitHub Actionsでの定期実行を想定。シグナルが「買い」または「売り」の場合のみ
Slackに通知します(様子見の場合は通知しません)。

【必要な環境変数】
  SLACK_WEBHOOK_URL : SlackのIncoming Webhook URL

【必要ライブラリ】
  pip install yfinance pandas numpy requests --break-system-packages
"""

import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime


def fetch_data(period="5d", interval="5m"):
    df = yf.download("^N225", period=period, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError("データ取得に失敗しました。市場が開いていない、もしくはネットワーク接続を確認してください。")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def add_indicators(df):
    close = df["Close"]

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

    return df


def judge_signal(row, prev_row):
    score = 0
    reasons = []

    if prev_row["MA5"] <= prev_row["MA25"] and row["MA5"] > row["MA25"]:
        score += 2
        reasons.append("MA5がMA25を上抜け(ゴールデンクロス)")
    elif prev_row["MA5"] >= prev_row["MA25"] and row["MA5"] < row["MA25"]:
        score -= 2
        reasons.append("MA5がMA25を下抜け(デッドクロス)")
    elif row["MA5"] > row["MA25"]:
        score += 0.5
        reasons.append("短期MAが長期MAの上(上昇トレンド継続)")
    else:
        score -= 0.5
        reasons.append("短期MAが長期MAの下(下降トレンド継続)")

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

    if score >= 2:
        signal = "買い"
    elif score <= -2:
        signal = "売り"
    else:
        signal = "様子見"

    return signal, score, reasons


def send_slack_notification(webhook_url, signal, score, reasons, latest, timestamp):
    emoji = "🟢" if signal == "買い" else "🔴" if signal == "売り" else "⚪"

    reasons_text = "\n".join(f"・{r}" for r in reasons)

    text = (
        f"{emoji} *マイクロ日経225 シグナル: {signal}* (スコア: {score:+.1f})\n"
        f"時刻: {timestamp}\n"
        f"日経平均: {latest['Close']:.2f}\n"
        f"MA5: {latest['MA5']:.2f} / MA25: {latest['MA25']:.2f}\n"
        f"RSI: {latest['RSI']:.1f}\n"
        f"\n根拠:\n{reasons_text}\n"
        f"\n※テクニカル指標による参考情報です。投資判断は自己責任で行ってください。"
    )

    payload = {"text": text}
    resp = requests.post(webhook_url, json=payload, timeout=10)
    resp.raise_for_status()


def main():
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")

    df = fetch_data(period="5d", interval="5m")
    df = add_indicators(df)
    df = df.dropna()

    if len(df) < 2:
        print("データが不足しています。")
        sys.exit(0)

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    timestamp = df.index[-1].strftime("%Y-%m-%d %H:%M")

    signal, score, reasons = judge_signal(latest, prev)

    print(f"[{timestamp}] 判定: {signal} (スコア: {score:+.1f})")
    for r in reasons:
        print(f"  - {r}")

    # 様子見の場合は通知しない
    if signal == "様子見":
        print("様子見のため通知はスキップします。")
        return

    if not webhook_url:
        print("警告: SLACK_WEBHOOK_URLが設定されていないため通知をスキップします。")
        return

    send_slack_notification(webhook_url, signal, score, reasons, latest, timestamp)
    print("Slackに通知しました。")


if __name__ == "__main__":
    main()
