import numpy as np
import pandas as pd
from .utils import ensure_schema

def backtest_fsm(df, p, stock_type="DEFAULT"):
    """
    回測引擎 - 有限狀態機版本
    
    Args:
        df: 包含指標和訊號的 DataFrame
        p: 參數字典
        stock_type: 股票類型 (WEIGHT/FINANCE/MOMENTUM)
    
    Returns:
        dict: 回測結果統計
    """
    df = df.copy().dropna(subset=["Close"])
    df = ensure_schema(df)

    if len(df) < 100:
        return {
            "df": df, 
            "trades": 0, 
            "total_return": 0.0, 
            "winrate": 0, 
            "profit_factor": 0, 
            "buy_reasons": [], 
            "sell_reasons": []
        }

    # 參數設定
    fee_buy = p["FEE_BUY"]
    fee_sell = p["FEE_SELL"]
    exit_cooldown = p.get("EXIT_COOLDOWN_DAYS", 5)  # 增加到 5 天
    
    # 提取陣列加速運算
    closes = df["Close"].values
    highs = df["High"].values
    buys = df["Buy_Signal"].values
    
    # 讀取 signals.py 產生的賣出訊號
    sells_all = df["Sell_Signal"].values
    sell_reasons_raw = df["Sell_Reason_Raw"].values
    buy_reasons_raw = df["Buy_Reason"].values

    n = len(df)
    
    # 初始化追蹤陣列
    equity = np.ones(n, dtype=float)
    pos_hist = np.zeros(n, dtype=int)
    
    # 狀態變數
    pos = 0              # 持倉狀態 (0=空手, 1=持有)
    entry_price = None   # 進場價格
    entry_idx = -1       # 進場索引
    last_exit_idx = -999 # 最後出場索引 (用於冷卻期)
    
    # 交易紀錄
    trades = []
    record_buy_reasons = []
    record_sell_reasons = []

    # 主迴圈 - 逐日模擬
    for i in range(1, n):
        equity[i] = equity[i-1]
        in_cooldown = (i - last_exit_idx) < exit_cooldown

        # ========== 持倉中 ==========
        if pos == 1:
            # 更新權益 (按收盤價計算)
            equity[i] *= (closes[i] / closes[i-1])

            # --- 賣出邏輯 (完全仰賴 signals.py) ---
            if sells_all[i]:
                # 執行賣出 (扣除手續費)
                equity[i] *= (1 - fee_sell)

                # 計算報酬
                raw_ret = (closes[i] / entry_price) - 1
                net_ret = (1 + raw_ret) * (1 - fee_buy) * (1 - fee_sell) - 1
                trades.append(net_ret)

                # 記錄賣出原因
                reason = sell_reasons_raw[i]
                record_sell_reasons.append(reason)

                # 清空持倉
                pos = 0
                entry_price = None
                entry_idx = -1
                last_exit_idx = i

        # ========== 空手中 ==========
        elif pos == 0:
            # --- 買進邏輯 ---
            if buys[i] and not in_cooldown:
                pos = 1
                entry_price = closes[i] * (1 + fee_buy)
                entry_idx = i
                equity[i] *= (1 - fee_buy)
                record_buy_reasons.append(buy_reasons_raw[i])

        pos_hist[i] = pos

    # 寫回 DataFrame
    df["Equity"] = equity
    df["Position"] = pos_hist

    # ========== 績效統計 ==========
    total_ret = equity[-1] - 1.0
    dd = (df["Equity"] / df["Equity"].cummax() - 1).min()

    # 獲利因子
    gross_profit = sum([t for t in trades if t > 0])
    gross_loss = abs(sum([t for t in trades if t < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    # 勝率
    winrate = np.mean([t > 0 for t in trades]) if trades else 0

    # Buy & Hold 報酬
    bh_return = (closes[-1] / closes[0]) - 1

    # 在市場時間比例
    in_market = (pos_hist == 1).mean()

    return {
        "df": df,
        "total_return": total_ret,
        "dd": dd,
        "trades": len(trades),
        "winrate": winrate,
        "bh_return": bh_return,
        "trades_list": trades,
        "profit_factor": pf,
        "in_market": in_market,
        "buy_reasons": record_buy_reasons,
        "sell_reasons": record_sell_reasons,
        "te": 0,      # Tracking Error (簡化)
        "sharpe": 0   # Sharpe Ratio (簡化)
    }
