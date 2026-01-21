# =========================================================
# Stock Tech Scorer Web App (V45.0 智慧賣出版)
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import altair as alt
import sys
import os
import importlib
import warnings

try:
    import twstock
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "twstock"])
    import twstock

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

st.set_page_config(page_title="阿勳簡易版股票分析", layout="wide", page_icon="📈")

sys.path.append(os.getcwd())
try:
    from stock_risk_tool import config, utils, indicators, signals, backtest, report
    importlib.reload(config)
except ImportError:
    st.error("❌ 找不到模組")
    st.stop()

def get_smart_name(ticker):
    code = ticker.split(".")[0]
    if code in twstock.codes:
        return f"{code} {twstock.codes[code].name}"
    return ticker

def explain_score_oldwang(row):
    breakdown = []
    if row.get('SanYang', False): breakdown.append(("✅ 三陽開泰", 20))
    if row.get('SiHai', False): breakdown.append(("✅ 四海遊龍", 20))
    if row.get('MA_Slope', 0) > config.P['MA_SLOPE_THRESHOLD']:
        breakdown.append(("✅ 趨勢陡峭", 10))

    # 賣出扣分
    reason = row.get('Sell_Reason_Raw', 'NONE')
    if reason == 'TAKE_PROFIT': breakdown.append(("⚠️ 高檔獲利訊號", -10))
    if reason == 'BIG_VOL': breakdown.append(("❌ 跌破爆量低點", -30))
    if reason == 'MA_BREAK': breakdown.append(("❌ 跌破生命線", -30))

    return breakdown

st.title("📈 阿勳簡易版股票分析 (V45.0 智慧賣出版)")
st.caption("🚀 策略升級：區分「高檔獲利了結(紫點)」與「防守停損(綠點)」，賣得更漂亮！")

st.sidebar.header("設定")
tickers_text = st.sidebar.text_area("股票代號", value=config.STOCK_LIST_TEXT.strip(), height=200)
start_date = st.sidebar.date_input("分析起始日", pd.to_datetime(config.P["START"]), disabled=True)
run_btn = st.sidebar.button("🚀 開始分析", type="primary")

if run_btn:
    input_list = tickers_text.split()
    monitor_list = [f"{x}.TW" if not x.endswith(".TW") else x for x in input_list if x.strip()]
    st.info(f"正在分析 {len(monitor_list)} 檔股票...")

    try:
        raw = yf.download(monitor_list, start=str(start_date), group_by='ticker', auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            price_fields = {"Open","High","Low","Close","Volume"}
            if len(set(raw.columns.get_level_values(0)) & price_fields) >= 4:
                raw = raw.swaplevel(axis=1).sort_index(axis=1)
    except Exception as e:
        st.error(f"下載失敗: {e}")
        st.stop()

    results = []
    stock_dfs = {}
    params = config.P

    bar = st.progress(0)

    for i, t in enumerate(monitor_list):
        bar.progress((i+1)/len(monitor_list))
        try:
            if t not in raw.columns:
                 if isinstance(raw.columns, pd.MultiIndex) and t in raw.columns.get_level_values(1):
                     df0 = raw.xs(t, level=1, axis=1)
                 else: continue
            else:
                 df0 = raw[t].copy() if isinstance(raw, pd.DataFrame) else raw[t]

            manual_type = config.TICKERS_CONFIG.get(t, "DEFAULT")

            df = utils.ensure_ohlcv(df0)
            df = utils.ensure_schema(df)
            df = indicators.add_indicators(df, params)
            df = signals.generate_signals(df, params, mode="OldWang", stock_type=manual_type)

            # --- 繪圖資料準備 ---
            pos = 0
            buy_markers = []
            sell_profit_markers = [] # 獲利了結點
            sell_stop_markers = []   # 停損點
            protect_markers = []

            buys = df["Buy_Signal"].values
            sells = df["Sell_Signal"].values
            reasons = df["Sell_Reason_Raw"].values
            protects = df["In_Protection"].values
            closes = df["Close"].values

            for k in range(len(df)):
                c_buy = np.nan
                c_sell_profit = np.nan
                c_sell_stop = np.nan
                c_protect = np.nan

                if pos == 0:
                    if buys[k]:
                        pos = 1
                        c_buy = closes[k]
                elif pos == 1:
                    if sells[k]:
                        pos = 0
                        # 區分賣出類型
                        if reasons[k] == "TAKE_PROFIT":
                            c_sell_profit = closes[k]
                        else:
                            c_sell_stop = closes[k]
                    elif protects[k]:
                        c_protect = closes[k]

                buy_markers.append(c_buy)
                sell_profit_markers.append(c_sell_profit)
                sell_stop_markers.append(c_sell_stop)
                protect_markers.append(c_protect)

            df["Buy_Marker"] = buy_markers
            df["Sell_Profit_Marker"] = sell_profit_markers
            df["Sell_Stop_Marker"] = sell_stop_markers
            df["Protect_Marker"] = protect_markers

            last_day = df.iloc[-1]
            score = last_day["Tech_Score"]
            score_details = explain_score_oldwang(last_day)

            if score >= 60: advice, color = "🔥 多頭", "red"
            elif score <= 30: advice, color = "❄️ 空頭", "green"
            else: advice, color = "⚠️ 震盪", "orange"

            display_name = get_smart_name(t)
            stop_col = "MA20"
            if manual_type == "MOMENTUM": stop_col = "MA10"
            elif manual_type == "FINANCE": stop_col = "MA60"

            results.append({
                "股票": t, "顯示名稱": display_name, "類別": manual_type,
                "技術評分": int(score), "評分細節": score_details,
                "建議": advice, "color": color,
                "最新收盤": f"{last_day['Close']:.1f}",
                "生命線": stop_col
            })
            stock_dfs[t] = df.tail(150)

        except Exception as e:
            st.warning(f"{t} 錯誤: {e}")

    bar.empty()

    if results:
        st.subheader("🔔 分析結果摘要")
        cols = st.columns(4)
        for idx, r in enumerate(results):
            with cols[idx % 4]:
                st.markdown(f"#### {r['顯示名稱']}")
                st.caption(f"策略: {r['類別']} | 生命線: {r['生命線']}")
                st.metric("技術評分", f"{r['技術評分']} 分")
                st.markdown(f"**狀態:** :{r['color']}[{r['建議']}]")
                with st.expander("📊 分數詳解"):
                    for item, pts in r['評分細節']:
                        st.write(f"{item} `{'+' if pts>0 else ''}{pts}`")
                st.divider()

        st.subheader("📈 賣出訊號詳解 (紫點vs綠點)")
        with st.expander("💡 賣出點怎麼看？", expanded=True):
             st.info("""
             * 🟣 **紫色點 (獲利了結)**：股價漲太多(乖離過大) + 跌破5日線。這是 **「賺錢賣」**，雖然趨勢還沒翻空，但先落袋為安。
             * 🟢 **綠色點 (防守停損)**：跌破生命線 或 爆量低點。這是 **「保命賣」**，趨勢已經轉弱，必須出場。
             * 🔵 **藍色點 (保護中)**：跌破生命線但RSI超賣，暫時不賣。
             """)

        tabs = st.tabs([r["顯示名稱"] for r in results])
        for idx, tab in enumerate(tabs):
            r = results[idx]
            t = r["股票"]
            stop_col = r["生命線"]
            df_plot = stock_dfs[t].copy().reset_index()
            if 'Date' not in df_plot.columns: df_plot.rename(columns={'index': 'Date'}, inplace=True)
            if 'Date' not in df_plot.columns: df_plot['Date'] = df_plot.iloc[:, 0]

            with tab:
                base = alt.Chart(df_plot).encode(x='Date:T')

                line = base.mark_line(color='#AAAAAA', strokeWidth=2).encode(
                    y=alt.Y('Close', scale=alt.Scale(zero=False), title='股價'),
                    tooltip=['Date', 'Close']
                )
                life_line = base.mark_line(color='#0000FF', strokeDash=[5, 5]).encode(y=stop_col)
                boom_line = base.mark_line(color='#800080', strokeWidth=2).encode(y='BigVol_Low')

                # 買點
                buy_points = base.mark_circle(color='red', size=100, opacity=1).encode(
                    y='Buy_Marker', tooltip=['Date', 'Close', alt.Tooltip('Buy_Marker', title='買入')]
                ).transform_filter(alt.datum.Buy_Marker > 0)

                # 獲利了結點 (紫)
                profit_points = base.mark_circle(color='#9932CC', size=100, opacity=1).encode(
                    y='Sell_Profit_Marker', tooltip=['Date', 'Close', alt.Tooltip('Sell_Profit_Marker', title='獲利了結')]
                ).transform_filter(alt.datum.Sell_Profit_Marker > 0)

                # 停損點 (綠)
                stop_points = base.mark_circle(color='green', size=100, opacity=1).encode(
                    y='Sell_Stop_Marker', tooltip=['Date', 'Close', alt.Tooltip('Sell_Stop_Marker', title='停損/破線')]
                ).transform_filter(alt.datum.Sell_Stop_Marker > 0)

                # 保護點 (藍)
                protect_points = base.mark_circle(color='blue', size=80, opacity=0.8).encode(
                    y='Protect_Marker', tooltip=['Date', 'Close', alt.Tooltip('Protect_Marker', title='保護中')]
                ).transform_filter(alt.datum.Protect_Marker > 0)

                chart = (line + life_line + boom_line + buy_points + profit_points + stop_points + protect_points).interactive()
                st.altair_chart(chart, use_container_width=True)

                # 乖離率圖 (輔助判斷獲利了結)
                st.markdown("##### 乖離率 (Bias) - 超過 15% 容易觸發獲利了結")
                bias_chart = base.mark_area(opacity=0.3, color='purple').encode(
                    y='Bias'
                )
                st.altair_chart(bias_chart.interactive(), use_container_width=True)

    else:
        st.warning("無數據")
