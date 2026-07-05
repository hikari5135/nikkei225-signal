# -*- coding: utf-8 -*-
"""
フォワードテスト & 時間帯別・曜日別分析ツール
================================================

signal_history.jsonl(実際にリアルタイムで出力されたシグナル)を元に、
「本当に出たシグナルが、その後どうなったか」を検証する。

過去データに同じロジックを再適用する backtest.py とは異なり、
このツールは実運用で実際に出力されたシグナルのみを対象にする
「フォワードテスト」であるため、より実態に近い精度の指標となる。

注意:
- 運用開始直後はデータ件数が少ないため、統計として安定するまで
  数週間程度かかる見込み。
- 曜日は日本語(月/火/水/木/金)で表示する。
"""

import os
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(BASE_DIR, "docs", "signal_history.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "docs", "forward_test.json")

HORIZONS_MINUTES = [30, 60, 120]
PRIMARY_HORIZON = 60  # 時間帯別・曜日別集計の代表値として使う時間軸
MIN_SAMPLES_FOR_STATS = 5  # この件数未満は「データ不足」として扱う

WEEKDAY_JP = ["月", "火", "水", "木", "金", "土", "日"]


def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    records = []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                r["time_dt"] = datetime.strptime(r["time"], "%Y-%m-%d %H:%M")
                records.append(r)
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
    records.sort(key=lambda r: r["time_dt"])
    return records


def find_price_at_or_after(records, target_time, start_index):
    for i in range(start_index, len(records)):
        if records[i]["time_dt"] >= target_time:
            return records[i]["close"], records[i]["time_dt"]
    return None, None


def evaluate_signal_outcome(records, idx, minutes):
    """指定したシグナルが、minutes分後にどうなったか(win/lose/flat)を判定する"""
    entry = records[idx]
    target_time = entry["time_dt"] + timedelta(minutes=minutes)
    future_close, _ = find_price_at_or_after(records, target_time, idx + 1)
    if future_close is None:
        return None

    change = future_close - entry["close"]
    signal = entry["signal"]

    if signal == "買い":
        outcome = "win" if change > 0 else ("lose" if change < 0 else "flat")
        directional_change = change
    else:  # 売り
        outcome = "win" if change < 0 else ("lose" if change > 0 else "flat")
        directional_change = -change

    return outcome, directional_change


def build_overall_stats(records, signal_indices):
    """時間軸ごとの全体統計(30分後/60分後/120分後)"""
    results = {}
    for minutes in HORIZONS_MINUTES:
        horizon_key = f"{minutes}min"
        stats = {
            "買い": {"win": 0, "lose": 0, "flat": 0, "total_change": 0.0},
            "売り": {"win": 0, "lose": 0, "flat": 0, "total_change": 0.0},
        }
        for idx in signal_indices:
            outcome_data = evaluate_signal_outcome(records, idx, minutes)
            if outcome_data is None:
                continue
            outcome, directional_change = outcome_data
            signal = records[idx]["signal"]
            stats[signal][outcome] += 1
            stats[signal]["total_change"] += directional_change

        horizon_result = {}
        for signal, bucket in stats.items():
            n = bucket["win"] + bucket["lose"] + bucket["flat"]
            win_rate = round(bucket["win"] / n * 100, 1) if n > 0 else None
            avg_change = round(bucket["total_change"] / n, 2) if n > 0 else None
            horizon_result[signal] = {
                "n": n,
                "win": bucket["win"],
                "lose": bucket["lose"],
                "flat": bucket["flat"],
                "win_rate": win_rate,
                "avg_change_points": avg_change,
                "sufficient_data": n >= MIN_SAMPLES_FOR_STATS,
            }
        results[horizon_key] = horizon_result
    return results


def build_time_breakdown(records, signal_indices):
    """時間帯別(0-23時)・曜日別(月-日)の成績を、代表時間軸(PRIMARY_HORIZON)で集計"""
    by_hour = defaultdict(lambda: {"win": 0, "lose": 0, "flat": 0})
    by_weekday = defaultdict(lambda: {"win": 0, "lose": 0, "flat": 0})

    for idx in signal_indices:
        outcome_data = evaluate_signal_outcome(records, idx, PRIMARY_HORIZON)
        if outcome_data is None:
            continue
        outcome, _ = outcome_data
        entry_time = records[idx]["time_dt"]
        hour = entry_time.hour
        weekday = WEEKDAY_JP[entry_time.weekday()]

        by_hour[hour][outcome] += 1
        by_weekday[weekday][outcome] += 1

    def summarize(bucket_dict, sort_keys):
        result = []
        for key in sort_keys:
            if key not in bucket_dict:
                continue
            b = bucket_dict[key]
            n = b["win"] + b["lose"] + b["flat"]
            win_rate = round(b["win"] / n * 100, 1) if n > 0 else None
            result.append({
                "label": key,
                "n": n,
                "win": b["win"],
                "lose": b["lose"],
                "win_rate": win_rate,
                "sufficient_data": n >= MIN_SAMPLES_FOR_STATS,
            })
        return result

    hour_labels = sorted(by_hour.keys())
    hour_result = summarize(by_hour, hour_labels)
    for r in hour_result:
        r["label"] = f"{r['label']}時"

    weekday_result = summarize(by_weekday, WEEKDAY_JP)

    return hour_result, weekday_result


def main():
    records = load_history()
    signal_indices = [i for i, r in enumerate(records) if r["signal"] in ("買い", "売り")]

    if len(records) < 2 or len(signal_indices) == 0:
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_history_points": len(records),
            "total_signals": len(signal_indices),
            "overall": {},
            "by_hour": [],
            "by_weekday": [],
            "note": "まだ十分なシグナル履歴がありません。運用を続けるとデータが蓄積されます。",
        }
    else:
        overall = build_overall_stats(records, signal_indices)
        by_hour, by_weekday = build_time_breakdown(records, signal_indices)
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_history_points": len(records),
            "total_signals": len(signal_indices),
            "overall": overall,
            "by_hour": by_hour,
            "by_weekday": by_weekday,
            "min_samples_for_stats": MIN_SAMPLES_FOR_STATS,
        }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"forward_test written: {OUTPUT_PATH}")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
