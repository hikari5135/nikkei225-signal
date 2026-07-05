# -*- coding: utf-8 -*-
"""
マイクロ日経225 デイトレシグナル バックテストツール(1日1回版)
================================================================

既存の backtest.py は「シグナルが変わるたびに何度でもドテンする」方式だが、
実際のデイトレ運用によりレベルに近い「1日の中で最初に出たシグナルだけを
採用し、その日のうちに手仕舞う」ルールで検証する。

ルール:
- その日の最初の「買い」または「売り」シグナルでのみエントリーする
- 同日中に追加のシグナルが出ても無視する(1日1トレードまで)
- その日の最終足(取引終了時点)で強制決済する
- シグナルが1度も出なかった日はトレードなし

既存の backtest.py・data.json・dashboard には一切影響しません。

【必要ライブラリ】
  pip install yfinance pandas numpy --break-system-packages
"""

import os
import sys
import json
from collections import defaultdict
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone


OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "backtest_daily_single_entry.json")


def fetch_data(period="60d", interval="5m"):
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
    """nikkei_daytrade_signal.py と同一ロジック"""
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


def run_daily_single_entry_backtest(df):
    """1日の中で最初のシグナルのみ採用し、その日のうちに手仕舞う"""
    trades = []
    rows = df.to_dict("records")
    times = df.index.tolist()

    # 日付ごとにその日のインデックス一覧をまとめる
    day_indices = defaultdict(list)
    for i, t in enumerate(times):
        day_indices[t.date()].append(i)

    for date in sorted(day_indices.keys()):
        indices = day_indices[date]
        position = None

        for idx in indices:
            if idx == 0:
                continue  # 直前の足が必要なため先頭は判定不可
            row = rows[idx]
            prev_row = rows[idx - 1]
            signal = judge_signal(row, prev_row)

            if position is None and signal in ("買い", "売り"):
                position = {
                    "side": "long" if signal == "買い" else "short",
                    "entry_price": row["Close"],
                    "entry_time": times[idx].strftime("%Y-%m-%d %H:%M"),
                }
                # この日は1トレードまでなので、エントリー後はその日の残りを見なくてよい
                break

        if position:
            last_idx = indices[-1]
            last_price = rows[last_idx]["Close"]
            last_time = times[last_idx].strftime("%Y-%m-%d %H:%M")
            if position["side"] == "long":
                pnl = last_price - position["entry_price"]
            else:
                pnl = position["entry_price"] - last_price
            trades.append({**position, "exit_price": last_price, "exit_time": last_time, "pnl": pnl})

    return trades


def summarize(trades):
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "max_win": 0,
            "max_loss": 0,
            "max_drawdown": 0,
            "expectancy": 0,
            "profit_factor": None,
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
    df, ticker = fetch_data(period="60d", interval="5m")
    df = add_indicators(df)
    df = df.dropna()

    if len(df) < 30:
        print("データが不足しています。")
        sys.exit(0)

    trades = run_daily_single_entry_backtest(df)
    summary = summarize(trades)
    equity_curve = build_equity_curve(trades)

    total_days = len(set(df.index.date))

    print("=" * 50)
    print(f"バックテスト結果(1日1回版) (データ元: {ticker}, 期間: {df.index[0]} 〜 {df.index[-1]})")
    print("=" * 50)
    print(f"対象営業日数: {total_days}日")
    print(f"総トレード数: {summary['total_trades']}(シグナルが出なかった日は0トレード)")
    print(f"勝率: {summary['win_rate']}% ({summary.get('win_trades', 0)}勝{summary.get('lose_trades', 0)}敗)")
    print(f"累計損益: {summary['total_pnl']:+.2f} pt")
    print(f"期待値(1トレード平均): {summary['expectancy']:+.2f} pt")

    result = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": ticker,
        "period_start": df.index[0].strftime("%Y-%m-%d %H:%M"),
        "period_end": df.index[-1].strftime("%Y-%m-%d %H:%M"),
        "total_days": total_days,
        "unit_note": "損益は日経225指数ポイント換算です(円換算・手数料/スプレッド控除前の理論値)。1日1回・その日最初のシグナルのみ採用し、当日中に強制決済するルールです。",
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
