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
ECONOMIC_EVENTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "economic_events.json")
EVENT_WARNING_BEFORE_MINUTES = 60   # イベント発表の何分前から警告を出すか
EVENT_WARNING_AFTER_MINUTES = 30    # イベント発表の何分後まで警告を続けるか


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
    breakdown = {"MA": 0, "RSI": 0, "MACD": 0, "BB": 0}

    if prev_row["MA5"] <= prev_row["MA25"] and row["MA5"] > row["MA25"]:
        score += 3
        breakdown["MA"] = 3
        reasons.append("MA5 crossed above MA25 (Golden Cross)")
    elif prev_row["MA5"] >= prev_row["MA25"] and row["MA5"] < row["MA25"]:
        score -= 3
        breakdown["MA"] = -3
        reasons.append("MA5 crossed below MA25 (Dead Cross)")

    if row["RSI"] < 30:
        score += 1.5
        breakdown["RSI"] = 1.5
        reasons.append(f"RSI={row['RSI']:.1f} (oversold)")
    elif row["RSI"] > 70:
        score -= 1.5
        breakdown["RSI"] = -1.5
        reasons.append(f"RSI={row['RSI']:.1f} (overbought)")

    if prev_row["MACD_hist"] <= 0 and row["MACD_hist"] > 0:
        score += 1.5
        breakdown["MACD"] = 1.5
        reasons.append("MACD histogram turned positive")
    elif prev_row["MACD_hist"] >= 0 and row["MACD_hist"] < 0:
        score -= 1.5
        breakdown["MACD"] = -1.5
        reasons.append("MACD histogram turned negative")

    if row["Close"] <= row["BB_lower"]:
        score += 1
        breakdown["BB"] = 1
        reasons.append("Price <= Bollinger -2sigma (possible rebound)")
    elif row["Close"] >= row["BB_upper"]:
        score -= 1
        breakdown["BB"] = -1
        reasons.append("Price >= Bollinger +2sigma (possible pullback)")

    if score >= 3:
        signal = "買い"
    elif score <= -3:
        signal = "売り"
    else:
        signal = "様子見"

    return signal, score, reasons, breakdown


def score_to_stars(score):
    """スコアを5段階の星評価に変換する"""
    abs_score = abs(score)
    if abs_score >= 6:
        stars = 5
    elif abs_score >= 4.5:
        stars = 4
    elif abs_score >= 3:
        stars = 3
    elif abs_score >= 1.5:
        stars = 2
    else:
        stars = 1

    if score >= 3:
        label = "強い買い" if stars >= 4 else "買い"
    elif score <= -3:
        label = "強い売り" if stars >= 4 else "売り"
    elif score > 0:
        label = "弱い買い"
    elif score < 0:
        label = "弱い売り"
    else:
        label = "様子見"

    return stars, label


def describe_current_state(row):
    """シグナルの有無にかかわらず、現在の各指標の状態を説明文にする"""
    lines = []

    if row["MA5"] > row["MA25"]:
        lines.append(f"MA5({row['MA5']:.0f})がMA25({row['MA25']:.0f})を上回っており、短期は上昇基調")
    else:
        lines.append(f"MA5({row['MA5']:.0f})がMA25({row['MA25']:.0f})を下回っており、短期は下降基調")

    rsi = row["RSI"]
    if rsi >= 70:
        lines.append(f"RSI={rsi:.1f}で過熱(買われすぎ)水準")
    elif rsi <= 30:
        lines.append(f"RSI={rsi:.1f}で売られすぎ水準")
    else:
        lines.append(f"RSI={rsi:.1f}で中立圏")

    if row["MACD_hist"] > 0:
        lines.append("MACDヒストグラムはプラス(上昇モメンタム)")
    else:
        lines.append("MACDヒストグラムはマイナス(下降モメンタム)")

    if row["Close"] >= row["BB_upper"]:
        lines.append("価格はボリンジャーバンド+2σ超で過熱気味")
    elif row["Close"] <= row["BB_lower"]:
        lines.append("価格はボリンジャーバンド-2σ未満で売られすぎ気味")
    else:
        lines.append("価格はボリンジャーバンドの通常レンジ内")

    return lines


def build_natural_summary(row, signal, composite, timeframes):
    """「今なぜこの判定なのか」を一文で要約する"""
    ma_trend = "上昇基調" if row["MA5"] > row["MA25"] else "下降基調"

    rsi = row["RSI"]
    if rsi >= 70:
        rsi_note = f"RSI({rsi:.0f})が過熱気味"
    elif rsi <= 30:
        rsi_note = f"RSI({rsi:.0f})が売られすぎ水準"
    else:
        rsi_note = f"RSI({rsi:.0f})は中立圏"

    macd_note = "MACDは上向き" if row["MACD_hist"] > 0 else "MACDは下向き"

    tf_1h = timeframes.get("1h", {}).get("signal") if timeframes else None
    tf_part = f" 1時間足では「{tf_1h}」。" if tf_1h else ""

    composite_part = f" 複合判定は「{composite}」。" if composite else ""

    return (
        f"短期は{ma_trend}、{macd_note}、{rsi_note}のため「{signal}」と判定。"
        f"{tf_part}{composite_part}"
    )


MAX_POSSIBLE_SCORE = 7.0  # MA(3)+RSI(1.5)+MACD(1.5)+BB(1) の理論上の最大値


def normalize_score(score):
    """スコアを-100〜+100の範囲に正規化する"""
    normalized = (score / MAX_POSSIBLE_SCORE) * 100
    normalized = max(-100, min(100, normalized))
    return round(normalized)


def calc_stop_for_side(side, close_price, atr, atr_mult=3.0):
    """指定した方向(buy/sell)を仮定した場合の損切りラインを計算する(現在のシグナルに関係なく計算)"""
    if atr is None or pd.isna(atr):
        return None
    if side == "buy":
        return round(float(close_price - atr * atr_mult), 2)
    elif side == "sell":
        return round(float(close_price + atr * atr_mult), 2)
    return None


def calc_tp_for_side(side, close_price, atr, atr_mult=3.0, rr=2.0):
    """指定した方向(buy/sell)を仮定した場合の利確目標を計算する"""
    if atr is None or pd.isna(atr):
        return None
    risk = atr * atr_mult
    if side == "buy":
        return round(float(close_price + risk * rr), 2)
    elif side == "sell":
        return round(float(close_price - risk * rr), 2)
    return None


def build_suggested_levels(close_price, atr):
    """買い/売りそれぞれを仮定した場合の推奨レベル(Entry/SL/TP1/TP2/RR)"""
    levels = {}
    for side in ("buy", "sell"):
        sl = calc_stop_for_side(side, close_price, atr)
        tp1 = calc_tp_for_side(side, close_price, atr, rr=1.5)
        tp2 = calc_tp_for_side(side, close_price, atr, rr=3.0)
        levels[side] = {
            "entry": round(float(close_price), 2),
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "rr1": 1.5,
            "rr2": 3.0,
        }
    return levels


def calc_daily_change(df):
    """直近の終値と、直近と異なる日付の最後の終値(前営業日終値)を比較する"""
    if df.empty:
        return None, None
    latest_close = float(df.iloc[-1]["Close"])
    latest_date = df.index[-1].date()

    prev_day_rows = df[df.index.date != latest_date]
    if prev_day_rows.empty:
        return None, None

    prev_close = float(prev_day_rows.iloc[-1]["Close"])
    if prev_close == 0:
        return None, None

    change = round(latest_close - prev_close, 2)
    change_pct = round((latest_close - prev_close) / prev_close * 100, 2)
    return change, change_pct


def calc_stop_price(signal, close_price, atr, atr_mult=3.0):
    if atr is None or pd.isna(atr):
        return None
    if signal == "買い":
        return round(float(close_price - atr * atr_mult), 2)
    elif signal == "売り":
        return round(float(close_price + atr * atr_mult), 2)
    return None


DEFAULT_RR = 2.0  # リスクリワード比率(デフォルト1:2)


def calc_take_profit(signal, close_price, atr, atr_mult=3.0, rr=DEFAULT_RR):
    """ATRベースの損切り幅(risk)にRR倍した値を利確目標(reward)とする"""
    if atr is None or pd.isna(atr):
        return None
    risk = atr * atr_mult
    if signal == "買い":
        return round(float(close_price + risk * rr), 2)
    elif signal == "売り":
        return round(float(close_price - risk * rr), 2)
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
    # DiscordのWebhookは "content" キー、Slackは "text" キーを使う。
    # URLに discord.com が含まれる場合は自動でDiscord向けの形式に切り替える。
    if "discord" in webhook_url:
        payload = {"content": text}
    else:
        payload = {"text": text}
    resp = requests.post(webhook_url, json=payload, timeout=10)
    resp.raise_for_status()


def build_json(df, signal, score, reasons, timestamp, ticker, stop_price, timeframes=None, composite=None, economic_events=None, breakdown=None, take_profit=None, stars=None, star_label=None, current_state=None, summary=None, atr=None, suggested_levels=None, daily_change=None, daily_change_pct=None):
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
        "score_normalized": normalize_score(score),
        "reasons": reasons,
        "stop_price": stop_price,
        "take_profit": take_profit,
        "risk_reward": DEFAULT_RR if (stop_price is not None and take_profit is not None) else None,
        "score_breakdown": breakdown or {},
        "stars": stars,
        "star_label": star_label,
        "current_state": current_state or [],
        "summary": summary,
        "atr": None if atr is None or pd.isna(atr) else round(float(atr), 2),
        "suggested_levels": suggested_levels or {},
        "daily_change": daily_change,
        "daily_change_pct": daily_change_pct,
        "candles": candles,
        "data_source": ticker,
        "timeframes": timeframes or {},
        "composite": composite,
        "economic_events": economic_events or {"upcoming_events": [], "warning": False, "warning_events": []},
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
        signal, score, _reasons, _breakdown = judge_signal(latest_row, prev_row)
        return {
            "label": label,
            "signal": signal,
            "score": score,
            "score_normalized": normalize_score(score),
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


def check_economic_events(now_jst):
    """経済指標カレンダーを確認し、直前・直後の警告と今後のイベント一覧を返す"""
    result = {"upcoming_events": [], "warning": False, "warning_events": []}

    if not os.path.exists(ECONOMIC_EVENTS_PATH):
        return result

    try:
        with open(ECONOMIC_EVENTS_PATH, "r", encoding="utf-8") as f:
            events = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"DEBUG economic_events load failed: {e}")
        return result

    upcoming = []
    warnings = []
    for ev in events:
        try:
            ev_time = datetime.strptime(ev["time"], "%Y-%m-%d %H:%M")
        except (KeyError, ValueError):
            continue

        diff_minutes = (ev_time - now_jst).total_seconds() / 60

        # 直前(EVENT_WARNING_BEFORE_MINUTES分前)〜直後(EVENT_WARNING_AFTER_MINUTES分後)なら警告対象
        if -EVENT_WARNING_AFTER_MINUTES <= diff_minutes <= EVENT_WARNING_BEFORE_MINUTES:
            warnings.append({"time": ev["time"], "label": ev["label"]})

        # 今後のイベントは一覧に含める(直近5件)
        if diff_minutes >= -EVENT_WARNING_AFTER_MINUTES:
            upcoming.append({
                "time": ev["time"],
                "label": ev["label"],
                "hours_until": round(diff_minutes / 60, 1),
            })

    upcoming.sort(key=lambda e: e["hours_until"])
    result["upcoming_events"] = upcoming[:5]
    result["warning"] = len(warnings) > 0
    result["warning_events"] = warnings
    return result


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

    signal, score, reasons, breakdown = judge_signal(latest, prev)
    stop_price = calc_stop_price(signal, latest["Close"], latest.get("ATR"))
    take_profit = calc_take_profit(signal, latest["Close"], latest.get("ATR"))
    stars, star_label = score_to_stars(score)
    current_state = describe_current_state(latest)

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

    economic_events = check_economic_events(df.index[-1].to_pydatetime().replace(tzinfo=None))
    if economic_events["warning"]:
        for ev in economic_events["warning_events"]:
            print(f"  WARNING: economic event nearby - {ev['time']} {ev['label']}")

    summary = build_natural_summary(latest, signal, composite, timeframes)
    print(f"  summary: {summary}")

    suggested_levels = build_suggested_levels(latest["Close"], latest.get("ATR"))
    daily_change, daily_change_pct = calc_daily_change(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    data = build_json(df, signal, score, reasons, timestamp, ticker, stop_price, timeframes, composite, economic_events, breakdown, take_profit, stars, star_label, current_state, summary, latest.get("ATR"), suggested_levels, daily_change, daily_change_pct)
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
