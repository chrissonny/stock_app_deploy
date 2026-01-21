import pandas as pd
import numpy as np

def build_suitability_report(results):
    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)

    report = []

    for _, row in df_res.iterrows():
        t = row["股票"]
        score = row.get("技術評分", 0) # 取得最新評分
        trades = row.get("交易筆數", 0)
        winrate = row.get("勝率", 0)
        pf = row.get("profit_factor", 0)
        total_ret = row.get("策略報酬", 0)

        # --- 診斷邏輯 (根據評分) ---
        status = ""
        note = []

        # 1. 評分判斷
        if score >= 70:
            status = "🔥 強力看漲"
        elif score >= 50:
            status = "✅ 偏多整理"
        elif score >= 30:
            status = "⚠️ 動能轉弱"
        else:
            status = "❌ 空頭走勢"

        # 2. 策略體質判斷
        if trades > 0 and pf < 1.0:
            note.append("歷史期望值低")
        if total_ret < -0.1:
            note.append("近期虧損中")

        final_note = "、".join(note) if note else "體質健康"

        report.append({
            "股票代號": t,
            "技術評分": int(score), # 顯示整數
            "狀態": status,
            "策略總報酬": f"{total_ret*100:.1f}%",
            "交易次數": trades,
            "PF(獲利因子)": f"{pf:.2f}",
            "勝率": f"{winrate*100:.0f}%",
            "診斷": final_note
        })

    return pd.DataFrame(report)
