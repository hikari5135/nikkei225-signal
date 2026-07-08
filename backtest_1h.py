# -*- coding: utf-8 -*-
"""
マイクロ日経225 デイトレシグナル バックテストツール(1時間足・長期版)
====================================================================

既存の backtest.py(5分足・直近60日)と同じロジックだが、
1時間足を使うことで約2年分のバックテストを行う。

注意:
- これは実際のライブシステム(5分足judge_signal)とは別の検証です。
  1時間足ベースでシグナルを再計算しているため、5分足の精度をそのまま
  代表するものではありません。あくまで「同じロジックを長期間・粗い足で
  見た場合の傾向」を確認するための参考値です。
- Yahoo Financeの仕様上、1時間足は直近730日程度まで取得可能です。

既存の backtest.py・data.json・dashboard には一切影響しません。

【必要ライブラリ】
  pip install yfinance pandas numpy --break-system-packages
"""

import os
import sys
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone


OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "backtest_1h.json")


def fetch_data(period="730d", interval="60m"):
    for ticker in ("NIY=F", "^N225"):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
        except Exception:
            df = None
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert("Asia/Tokyo")
            else:
                df.index = df.index.tz_convert("Asia/Tokyo")
            print(f"データ取得元: {ticker} ({len(df)}本)")
            return df, ticker
    raise RuntimeError("データ取得に失敗しました。")


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
    """nikkei_daytrade_signal.py と同一ロジック(1時間足に適用)"""
    score = 0

    if prev_row["MA5"] <= prev_row["MA25"] and row["MA5"] > row["MA25"]:
        score += 3
    elif prev_row["MA5"] >= prev_row["MA25"] and row["MA5"] < row["MA25"]:
        score -= 3

    if row["RSI"] < 30:
        score += 1.5
    elif row["RSI"] > 70:
        score -= 1.5

    if prev_row["MACD_hist"] <= 0 and row["MACD_hist"] > 0:
        score += 1.5
    elif prev_row["MACD_hist"] >= 0 and row["MACD_hist"] < 0:
        score -= 1.5

    if row["Close"] <= row["BB_lower"]:
        score += 1
    elif row["Close"] >= row["BB_upper"]:
        score -= 1

    if score >= 3:
        return "買い"
    elif score <= -3:
        return "売り"
    return "様子見"


def run_backtest(df):
    trades = []
    position = None

    rows = df.to_dict("records")
    times = df.index.tolist()

    for i in range(1, len(rows)):
        row = rows[i]
        prev_row = rows[i - 1]
        signal = judge_signal(row, prev_row)
        price = row["Close"]
        time_str = times[i].strftime("%Y-%m-%d %H:%M")

        if signal == "買い":
            if position and position["side"] == "short":
                pnl = position["entry_price"] - price
                trades.append({**position, "exit_price": price, "exit_time": time_str, "pnl": pnl})
                position = None
            if position is None:
                position = {"side": "long", "entry_price": price, "entry_time": time_str}

        elif signal == "売り":
            if position and position["side"] == "long":
                pnl = price - position["entry_price"]
                trades.append({**position, "exit_price": price, "exit_time": time_str, "pnl": pnl})
                position = None
            if position is None:
                position = {"side": "short", "entry_price": price, "entry_time": time_str}

    if position:
        last_price = rows[-1]["Close"]
        last_time = times[-1].strftime("%Y-%m-%d %H:%M")
        if position["side"] == "long":
            pnl = last_price - position["entry_price"]
        else:
            pnl = position["entry_price"] - last_price
        trades.append({**position, "exit_price": last_price, "exit_time": last_time, "pnl": pnl, "forced_exit": True})

    return trades


def summarize(trades):
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0, "total_pnl": 0,
            "avg_win": 0, "avg_loss": 0, "max_win": 0, "max_loss": 0,
            "max_drawdown": 0, "expectancy": 0, "profit_factor": None,
        }

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    drawdown = cum - peak
    max_dd = float(drawdown.min()) if len(drawdown) else 0

    total_win = float(sum(wins)) if wins else 0.0
    total_loss = abs(float(sum(losses))) if losses else 0.0
    profit_factor = round(total_win / total_loss, 2) if total_loss > 0 else None

    return {
        "total_trades": len(trades),
        "win_trades": len(wins),
        "lose_trades": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "total_pnl": round(float(sum(pnls)), 2),
        "avg_win": round(float(np.mean(wins)), 2) if wins else 0,
        "avg_loss": round(float(np.mean(losses)), 2) if losses else 0,
        "max_win": round(float(max(pnls)), 2),
        "max_loss": round(float(min(pnls)), 2),
        "max_drawdown": round(max_dd, 2),
        "expectancy": round(float(sum(pnls)) / len(trades), 2) if trades else 0,
        "profit_factor": profit_factor,
    }


def build_equity_curve(trades):
    if not trades:
        return []
    pnls = [t["pnl"] for t in trades]
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    drawdown = cum - peak

    curve = []
    for i, t in enumerate(trades):
        curve.append({
            "trade_no": i + 1,
            "exit_time": t["exit_time"],
            "cumulative_pnl": round(float(cum[i]), 2),
            "drawdown": round(float(drawdown[i]), 2),
        })
    return curve


def main():
    df, ticker = fetch_data(period="730d", interval="60m")
    df = add_indicators(df)
    df = df.dropna()

    if len(df) < 30:
        print("データが不足しています。")
        sys.exit(0)

    trades = run_backtest(df)
    summary = summarize(trades)
    equity_curve = build_equity_curve(trades)

    print("=" * 50)
    print(f"バックテスト結果(1時間足・長期版) (データ元: {ticker}, 期間: {df.index[0]} 〜 {df.index[-1]})")
    print("=" * 50)
    print(f"総トレード数: {summary['total_trades']}")
    print(f"勝率: {summary['win_rate']}% ({summary.get('win_trades', 0)}勝{summary.get('lose_trades', 0)}敗)")
    print(f"累計損益: {summary['total_pnl']:+.2f} pt")
    print(f"期待値(1トレード平均): {summary['expectancy']:+.2f} pt")
    print(f"プロフィットファクター: {summary['profit_factor']}")

    result = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": ticker,
        "period_start": df.index[0].strftime("%Y-%m-%d %H:%M"),
        "period_end": df.index[-1].strftime("%Y-%m-%d %H:%M"),
        "unit_note": "1時間足を使った長期検証です(約2年分)。実際のライブシステムは5分足でシグナル判定しているため、この結果は5分足の精度をそのまま代表するものではなく、同じロジックを長期間・粗い足で見た場合の参考値です。損益は日経225指数ポイント換算(円換算・手数料/スプレッド控除前の理論値)。",
        "summary": summary,
        "trades": trades[-50:],
        "equity_curve": equity_curve[-200:],
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n結果を書き出しました: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
