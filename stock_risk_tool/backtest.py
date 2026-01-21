import numpy as np
import pandas as pd
from .utils import ensure_schema

def backtest_fsm(df, p, stock_type="DEFAULT"):
    df = df.copy().dropna(subset=["Close"])
    df = ensure_schema(df)

    if len(df) < 100:
        return {"df": df, "trades": 0, "total_return": 0.0, "winrate": 0, "profit_factor": 0, "buy_reasons": [], "sell_reasons": []}

    fee_buy, fee_sell = p["FEE_BUY"], p["FEE_SELL"]
    closes = df["Close"].values
    highs = df["High"].values
    buys = df["Buy_Signal"].values

    # 讀取 signals.py 產生的賣出訊號
    sells_all = df["Sell_Signal"].values
    sell_reasons_raw = df["Sell_Reason_Raw"].values
    buy_reasons_raw = df["Buy_Reason"].values

    exit_cooldown = p.get("EXIT_COOLDOWN_DAYS", 3)

    n = len(df)
    equity = np.ones(n, dtype=float)
    pos_hist = np.zeros(n, dtype=int)
    pos = 0; entry_price = None; trades = []
    entry_idx = -1
    last_exit_idx = -999

    record_buy_reasons = []
    record_sell_reasons = []

    for i in range(1, n):
        equity[i] = equity[i-1]
        in_cooldown = (i - last_exit_idx) < exit_cooldown

        if pos == 1: # 持有中
            equity[i] *= (closes[i] / closes[i-1])

            # --- 賣出邏輯 (完全聽命於 signals.py) ---
            if sells_all[i]:
                # 執行賣出
                equity[i] *= (1 - fee_sell)

                # 結算損益
                raw_ret = (closes[i] / entry_price) - 1
                net_ret = (1 + raw_ret) * (1 - fee_buy) * (1 - fee_sell) - 1
                trades.append(net_ret)

                # 記錄原因
                reason = sell_reasons_raw[i]
                record_sell_reasons.append(reason)

                pos = 0; entry_price = None; entry_idx = -1
                last_exit_idx = i

        elif pos == 0: # 空手
            # --- 買進邏輯 ---
            if buys[i] and not in_cooldown:
                pos = 1
                entry_price = closes[i] * (1 + fee_buy)
                entry_idx = i
                equity[i] *= (1 - fee_buy)
                record_buy_reasons.append(buy_reasons_raw[i])

        pos_hist[i] = pos

    df["Equity"] = equity
    df["Position"] = pos_hist

    total_ret = equity[-1] - 1.0
    dd = (df["Equity"] / df["Equity"].cummax() - 1).min()

    gross_profit = sum([t for t in trades if t > 0])
    gross_loss = abs(sum([t for t in trades if t < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    return {
        "df": df, "total_return": total_ret, "dd": dd, "trades": len(trades),
        "winrate": np.mean([t > 0 for t in trades]) if trades else 0,
        "bh_return": (closes[-1] / closes[0]) - 1,
        "trades_list": trades, "profit_factor": pf,
        "in_market": (pos_hist == 1).mean(),
        "buy_reasons": record_buy_reasons, "sell_reasons": record_sell_reasons,
        "te": 0, "sharpe": 0 # 簡化回傳
    }
