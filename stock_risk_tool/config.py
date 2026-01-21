# =========================================================
# [老王實戰版 V3 - 參數補完] config.py
# 核心升級：補上 MA_SLOPE_THRESHOLD (斜率濾網) 避免報錯
# =========================================================
from datetime import datetime, timedelta

# 自動抓 3 年資料 (確保年線畫得出來)
today = datetime.today()
start_date_obj = today - timedelta(days=1095)
START_DATE_STR = start_date_obj.strftime("%Y-%m-%d")
END_DATE_STR = today.strftime("%Y-%m-%d")

STOCK_LIST_TEXT = """
2330
2317
2344
2454
2603
2881
2882
2449
"""

DEFAULT_TICKERS = [
    f"{x}.TW" if not x.endswith(".TW") else x
    for x in STOCK_LIST_TEXT.split()
    if x.strip()
]

TICKERS_CONFIG = {
    "2330.TW": "WEIGHT",
    "2317.TW": "WEIGHT",
    "2881.TW": "FINANCE",
    "2882.TW": "FINANCE",
    "2449.TW": "MOMENTUM",
}

P = {
    "START": START_DATE_STR,
    "END":   END_DATE_STR,

    "MA5": 5, "MA10": 10, "MA20": 20, "MA60": 60, "MA240": 240,

    # --- [關鍵修復] 必須包含這行 ---
    "MA_SLOPE_THRESHOLD": 0.005,  # 斜率門檻 (0.5%)

    "BIGVOL_MULT": 2.0,       # 爆大量倍數
    "BIG_RED_BODY_PCT": 0.03, # 長紅漲幅
    "GAP_UP_PCT": 0.005,      # 跳空幅度

    "SCORE_BUY_THRESHOLD": 70,
    "SCORE_SELL_THRESHOLD": 30,
    "FEE_BUY": 0.001425, "FEE_SELL": 0.004425,
    "STOP_BUFFER_PCT": 0.015,
    "BIGVOL_VALID_DAYS": 60,
    "VOL_MA": 20, "ATR_N": 14, "KD_N": 9, "RSI_N": 14,
    "MACD_FAST": 12, "MACD_SLOW": 26, "MACD_SIGNAL": 9
}
