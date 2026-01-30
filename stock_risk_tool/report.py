import pandas as pd
import numpy as np

def build_suitability_report(results):
    """
    建立適用性分析報告
    
    根據回測結果產生診斷報告,包含:
    - 技術評分
    - 狀態判斷
    - 策略績效
    - 診斷建議
    
    Args:
        results: 回測結果列表 (每個元素為一檔股票的結果字典)
    
    Returns:
        DataFrame: 格式化的報告表格
    """
    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)

    report = []

    for _, row in df_res.iterrows():
        ticker = row["股票"]
        score = row.get("技術評分", 0)          # 取得最新技術評分
        trades = row.get("交易筆數", 0)
        winrate = row.get("勝率", 0)
        pf = row.get("profit_factor", 0)
        total_ret = row.get("策略報酬", 0)

        # ========================================================
        # 診斷邏輯 (根據評分與績效)
        # ========================================================
        status = ""
        notes = []

        # 1. 技術評分判斷
        if score >= 70:
            status = "🔥 強力看漲"
        elif score >= 50:
            status = "✅ 偏多整理"
        elif score >= 30:
            status = "⚠️ 動能轉弱"
        else:
            status = "❌ 空頭走勢"

        # 2. 策略體質判斷
        if trades > 0:
            if pf < 1.0:
                notes.append("歷史期望值低")
            elif pf > 2.0:
                notes.append("策略效果良好")
        
        if total_ret < -0.1:
            notes.append("近期虧損中")
        elif total_ret > 0.2:
            notes.append("績效優異")
        
        # 3. 勝率判斷
        if trades > 0:
            if winrate < 0.4:
                notes.append("勝率偏低")
            elif winrate > 0.6:
                notes.append("勝率健康")

        # 綜合診斷
        final_note = "、".join(notes) if notes else "體質健康"

        # 加入報告
        report.append({
            "股票代號": ticker,
            "技術評分": int(score),  # 顯示整數
            "狀態": status,
            "策略總報酬": f"{total_ret*100:.1f}%",
            "交易次數": trades,
            "PF(獲利因子)": f"{pf:.2f}",
            "勝率": f"{winrate*100:.0f}%",
            "診斷": final_note
        })

    return pd.DataFrame(report)


def generate_detailed_stats(result):
    """
    產生單一股票的詳細統計報告
    
    Args:
        result: 單一股票的回測結果字典
    
    Returns:
        dict: 詳細統計資訊
    """
    trades_list = result.get("trades_list", [])
    
    if not trades_list:
        return {
            "總交易次數": 0,
            "平均報酬": 0,
            "最大單筆獲利": 0,
            "最大單筆虧損": 0,
            "連續獲利次數": 0,
            "連續虧損次數": 0
        }
    
    trades_array = np.array(trades_list)
    
    # 基本統計
    total_trades = len(trades_list)
    avg_return = trades_array.mean()
    max_win = trades_array.max()
    max_loss = trades_array.min()
    
    # 連續統計
    winning_streak = 0
    losing_streak = 0
    current_win_streak = 0
    current_lose_streak = 0
    
    for trade in trades_list:
        if trade > 0:
            current_win_streak += 1
            current_lose_streak = 0
            winning_streak = max(winning_streak, current_win_streak)
        else:
            current_lose_streak += 1
            current_win_streak = 0
            losing_streak = max(losing_streak, current_lose_streak)
    
    return {
        "總交易次數": total_trades,
        "平均報酬": f"{avg_return*100:.2f}%",
        "最大單筆獲利": f"{max_win*100:.2f}%",
        "最大單筆虧損": f"{max_loss*100:.2f}%",
        "最大連續獲利": winning_streak,
        "最大連續虧損": losing_streak
    }


def format_summary_table(results):
    """
    格式化為簡潔的摘要表格
    
    Args:
        results: 回測結果列表
    
    Returns:
        DataFrame: 摘要表格
    """
    if not results:
        return pd.DataFrame()
    
    summary = []
    
    for r in results:
        summary.append({
            "代號": r.get("股票", "N/A"),
            "評分": int(r.get("技術評分", 0)),
            "策略報酬": f"{r.get('策略報酬', 0)*100:.1f}%",
            "B&H報酬": f"{r.get('bh_return', 0)*100:.1f}%",
            "交易": r.get("交易筆數", 0),
            "勝率": f"{r.get('勝率', 0)*100:.0f}%",
            "PF": f"{r.get('profit_factor', 0):.2f}"
        })
    
    return pd.DataFrame(summary)
