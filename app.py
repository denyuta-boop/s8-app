import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import plotly.graph_objects as go
import itertools

# --- ページ設定 ---
st.set_page_config(page_title="S6戦略 自動最適化ツール", layout="wide")

# --- 定数 ---
TICKER_MAP = {
    "USDJPY": "USDJPY=X", "MXNJPY": "MXNJPY=X", "PLNJPY": "PLNJPY=X",
    "CZKJPY": "CZKJPY=X", "CHFJPY": "CHFJPY=X", "ZARJPY": "ZARJPY=X",
    "TRYJPY": "TRYJPY=X", "EURJPY": "EURJPY=X"
}

DEFAULT_SWAP = {
    "MXNJPY": 15.5, "PLNJPY": 42.0, "ZARJPY": 16.1, "TRYJPY": 30.1,
    "CZKJPY": 10.0,
    "USDJPY": -150.0, "CHFJPY": 15.0, "EURJPY": -100.0
}

DEFAULT_LOT_SIZE = {
    "MXNJPY": 10000, "PLNJPY": 10000, "CZKJPY": 10000, "ZARJPY": 10000,
    "TRYJPY": 10000, "USDJPY": 10000, "CHFJPY": 10000, "EURJPY": 10000
}

# --- 関数定義 ---

@st.cache_data(ttl=3600)
def fetch_data(days=1095):
    """データ取得 (USDJPY VIP分離取得版)"""
    debug_logs = []
    
    try:
        # 1. USDJPYを単独で確実に取る (VIP待遇)
        usd_symbol = "USDJPY=X"
        other_symbols = [v for k, v in TICKER_MAP.items() if v != usd_symbol]
        
        # --- USDJPY 取得 ---
        data_usd = yf.download(usd_symbol, period=f"{days}d", progress=False, auto_adjust=False)
        
        # USDJPYの整形
        df_usd_clean = pd.DataFrame()
        if not data_usd.empty:
            if isinstance(data_usd, pd.DataFrame):
                target_col = None
                if 'Close' in data_usd.columns:
                    target_col = data_usd['Close']
                elif 'Adj Close' in data_usd.columns:
                    target_col = data_usd['Adj Close']
                elif isinstance(data_usd.columns, pd.MultiIndex):
                     try:
                         target_col = data_usd.xs('Close', axis=1, level=0).iloc[:, 0]
                     except:
                         target_col = data_usd.iloc[:, 0]
                else:
                    target_col = data_usd.iloc[:, 0]
                
                if target_col is not None:
                    df_usd_clean["USDJPY"] = target_col

        if df_usd_clean.empty:
            debug_logs.append("Critical Error: Failed to fetch USDJPY standalone.")
            return None, {}, None, debug_logs

        # --- その他通貨 取得 ---
        data_others = yf.download(other_symbols, period=f"{days}d", progress=False, auto_adjust=False)
        
        # その他通貨の整形
        df_others_clean = pd.DataFrame()
        if not data_others.empty:
            df_temp = pd.DataFrame()
            if isinstance(data_others.columns, pd.MultiIndex):
                if 'Close' in data_others.columns.get_level_values(0):
                    df_temp = data_others['Close'].copy()
                elif 'Adj Close' in data_others.columns.get_level_values(0):
                    df_temp = data_others['Adj Close'].copy()
                else:
                    df_temp = data_others.copy()
                    df_temp.columns = df_temp.columns.droplevel(0)
            else:
                if 'Close' in data_others.columns:
                     df_temp = data_others[['Close']].copy()
                else:
                     df_temp = data_others.copy()

            for col in df_temp.columns:
                col_str = str(col).upper()
                matched_name = None
                for internal_name, yahoo_symbol in TICKER_MAP.items():
                    if internal_name == "USDJPY": continue
                    search_key = yahoo_symbol.upper().replace("=X", "")
                    if search_key in col_str:
                        matched_name = internal_name
                        break
                if matched_name:
                    df_others_clean[matched_name] = df_temp[col]

        # --- 合体 (Merge) ---
        final_df = df_usd_clean.join(df_others_clean, how='outer')
        
        if final_df.empty: return None, {}, None, debug_logs
        
        final_df = final_df.dropna(axis=1, how='all')
        df_filled = final_df.ffill().bfill()
        df_filled = df_filled.dropna(how='all')

        if len(df_filled) < 10: return None, {}, None, debug_logs

        latest_rates = df_filled.iloc[-1].to_dict()
        returns = np.log(df_filled).diff().dropna()
        
        return returns, latest_rates, df_filled, debug_logs

    except Exception as e:
        debug_logs.append(f"Fatal Error: {str(e)}")
        return None, {}, None, debug_logs

def calculate_beta(asset_returns, benchmark_returns):
    common_idx = asset_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 10: return 0.0
    y = asset_returns.loc[common_idx]
    x = benchmark_returns.loc[common_idx]
    if x.std() == 0 or y.std() == 0: return 0.0
    slope, _, _, _, _ = stats.linregress(x, y)
    if np.isnan(slope): return 0.0
    return slope

def generate_weights(n):
    weights = []
    if n == 1:
        return [{0: 1.0}]
    elif n == 2:
        for i in range(1, 10): weights.append({0: i/10, 1: (10-i)/10})
    elif n == 3:
        for i in range(1, 9):
            for j in range(1, 9-i):
                k = 10 - i - j
                if k > 0: weights.append({0: i/10, 1: j/10, 2: k/10})
    elif n == 4:
        for i in range(1, 8):
            for j in range(1, 8-i):
                for k in range(1, 8-i-j):
                    l = 10 - i - j - k
                    if l > 0:
                        weights.append({0: i/10, 1: j/10, 2: k/10, 3: l/10})
    return weights

# --- サイドバー設定 ---
with st.sidebar:
    st.header("⚙️ 設定パネル")
    
    password = st.text_input("🔑 パスワード (未入力でデモモード)", type="password")
    
    if password == "s6secret":
        is_demo_mode = False
        st.success("🔓 認証成功: フル機能モード")
    else:
        is_demo_mode = True
        st.info("👀 デモモードで動作中 (通貨ペア制限あり)")

    capital = st.number_input("💰 運用資金 (円)", value=1000000, step=100000)
    leverage = st.number_input("⚙️ 目標レバレッジ (倍)", value=16.0, step=0.1)

    with st.expander("📝 スワップポイント設定", expanded=False):
        swap_inputs = {}
        for ccy, val in DEFAULT_SWAP.items():
            swap_inputs[ccy] = st.number_input(f"{ccy}", value=float(val), step=0.1)

    st.markdown("---")
    st.subheader("🛡️ リスク制御")
    
    calc_period_option = st.selectbox(
        "📊 β・相関の計算期間", 
        ["直近1年 (推奨)", "直近2年", "直近3年"], 
        index=0
    )
    
    target_beta = st.slider("許容するβの範囲 (±)", 0.01, 0.50, 0.05, step=0.01)
    target_corr = st.slider("最低相関係数", -1.0, 1.0, 0.80, step=0.05)
    
    st.caption("通貨保有比率の制限")
    other_limit = st.slider("🌍 TRY以外の最大比率制限 (%)", 10, 100, 40, step=10)
    try_limit = st.slider("🇹🇷 TRYJPYの最大比率制限 (%)", 0, 100, 20, step=5)
    
    st.subheader("🔢 構成通貨数")
    buy_count_range = st.slider("買い通貨ペア数 (範囲)", 1, 4, (1, 4))
    sell_count_range = st.slider("売り通貨ペア数 (範囲)", 1, 4, (1, 3))

    st.markdown("---")
    st.subheader("📈 グラフ表示設定")
    plot_period_option = st.radio(
        "バックテスト表示期間", 
        ["直近1年", "直近2年", "直近3年 (全期間)"], 
        index=0
    )

# --- メイン画面 ---
st.title("📱 S6戦略 自動最適化ツール")

if is_demo_mode:
    st.warning("🚧 現在は**デモモード**です。選択できる通貨ペアが制限されています。フル機能を使うにはサイドバーでパスワードを入力してください。")
    buy_options = ["MXNJPY", "TRYJPY"]
    buy_default = ["MXNJPY", "TRYJPY"]
    sell_options = ["USDJPY"]
    sell_default = ["USDJPY"]
else:
    buy_options = ["MXNJPY", "ZARJPY", "PLNJPY", "TRYJPY", "CZKJPY"]
    buy_default = ["MXNJPY", "ZARJPY", "PLNJPY", "TRYJPY", "CZKJPY"]
    sell_options = ["USDJPY", "CHFJPY", "EURJPY"]
    sell_default = ["USDJPY", "CHFJPY", "EURJPY"]

col1, col2 = st.columns(2)
with col1:
    buy_candidates = st.multiselect("📈 買い候補", options=buy_options, default=buy_default)
with col2:
    sell_candidates = st.multiselect("📉 売り候補", options=sell_options, default=sell_default)

# 計算ボタン処理
if st.button("🚀 計算スタート", type="primary"):
    
    if len(buy_candidates) < buy_count_range[0] or len(sell_candidates) < sell_count_range[0]:
        st.error("⚠️ エラー: 候補通貨の数が、指定された最小構成数より少ないです。")
        st.stop()

    with st.spinner("⏳ データ取得＆最適化計算中..."):
        df_full, current_rates, df_prices, debug_logs = fetch_data(days=1095)
        
        if df_full is None or df_full.empty:
            st.error("❌ データ取得エラー。以下のデバッグ情報を確認してください。")
            with st.expander("🛠️ デバッグ情報"):
                for log in debug_logs:
                    st.write(log)
        else:
            if "1年" in calc_period_option:
                calc_days = 250
            elif "2年" in calc_period_option:
                calc_days = 500
            else:
                calc_days = 750
            
            df_calc = df_full.tail(calc_days)
            
            if "USDJPY" not in df_calc.columns:
                st.error(f"❌ USDJPYのデータ取得に失敗しました。時間をおいて再試行してください。(取得できた列: {list(df_calc.columns)})")
                with st.expander("🛠️ デバッグ情報"):
                    for log in debug_logs:
                        st.write(log)
                st.stop()

            betas = {}
            for col in df_calc.columns:
                if col == "USDJPY": betas[col] = 1.0
                else: betas[col] = calculate_beta(df_calc[col], df_calc["USDJPY"])
            
            target_notional = capital * leverage
            valid_plans = []
            fallback_plans = [] # ★敗者復活用のリスト

            # --- 組み合わせ生成 & 事前計算 ---
            # 1. 買い
            buy_precalc = []
            for size in range(buy_count_range[0], min(buy_count_range[1], len(buy_candidates)) + 1):
                for combo in itertools.combinations(buy_candidates, size):
                    if not all(ccy in betas for ccy in combo): continue
                    weights_list = generate_weights(size)
                    for wp in weights_list:
                        pattern = {combo[i]: wp[i] for i in range(size)}
                        
                        is_valid_weight = True
                        for ccy, weight in pattern.items():
                            if ccy == "TRYJPY":
                                if weight > (try_limit / 100): is_valid_weight = False; break
                            else:
                                if weight > (other_limit / 100): is_valid_weight = False; break
                        if not is_valid_weight: continue

                        b_beta = sum(betas.get(ccy, 0) * w for ccy, w in pattern.items())
                        b_series = pd.Series(0.0, index=df_calc.index)
                        for ccy, w in pattern.items(): b_series += df_calc[ccy] * w
                        
                        daily_swap_buy = 0
                        valid_swap = True
                        side_notional = target_notional / 2
                        try:
                            for ccy, w in pattern.items():
                                rate = current_rates.get(ccy, 0)
                                if rate == 0: valid_swap = False; break
                                lots = (side_notional * w) / (rate * DEFAULT_LOT_SIZE[ccy])
                                daily_swap_buy += lots * swap_inputs.get(ccy, 0)
                        except: valid_swap = False
                        
                        if valid_swap:
                            buy_precalc.append({
                                "pattern": pattern, "beta": b_beta, "series": b_series, "swap": daily_swap_buy
                            })

            # 2. 売り
            sell_precalc = []
            for size in range(sell_count_range[0], min(sell_count_range[1], len(sell_candidates)) + 1):
                for combo in itertools.combinations(sell_candidates, size):
                    if not all(ccy in betas for ccy in combo): continue
                    weights_list = generate_weights(size)
                    for wp in weights_list:
                        pattern = {combo[i]: wp[i] for i in range(size)}
                        s_beta = sum(betas.get(ccy, 0) * w for ccy, w in pattern.items()) * -1
                        s_series = pd.Series(0.0, index=df_calc.index)
                        for ccy, w in pattern.items(): s_series += df_calc[ccy] * w
                        
                        daily_swap_sell = 0
                        valid_swap = True
                        side_notional = target_notional / 2
                        try:
                            for ccy, w in pattern.items():
                                rate = current_rates.get(ccy, 0)
                                if rate == 0: valid_swap = False; break
                                lots = (side_notional * w) / (rate * DEFAULT_LOT_SIZE[ccy])
                                daily_swap_sell += lots * swap_inputs.get(ccy, 0)
                        except: valid_swap = False

                        if valid_swap:
                            sell_precalc.append({
                                "pattern": pattern, "beta": s_beta, "series": s_series, "swap": daily_swap_sell
                            })

            # 3. マッチング
            for b_item in buy_precalc:
                for s_item in sell_precalc:
                    net_beta = b_item["beta"] + s_item["beta"]
                    
                    corr = b_item["series"].corr(s_item["series"])
                    if np.isnan(corr): corr = 0
                    
                    total_swap = b_item["swap"] + s_item["swap"]
                    
                    plan_data = {
                        "buy": b_item["pattern"], "sell": s_item["pattern"],
                        "beta": net_beta, "swap": total_swap, "corr": corr
                    }

                    if abs(net_beta) < target_beta and corr > target_corr:
                        valid_plans.append(plan_data)
                    else:
                        fallback_plans.append(plan_data)

            # 結果の選定
            final_best = None
            is_fallback = False

            if valid_plans:
                valid_plans.sort(key=lambda x: x["swap"], reverse=True)
                final_best = valid_plans[0]
            elif fallback_plans:
                # ★修正: βの絶対値が小さい(0に近い)順に並べ替え、第2キーでスワップ
                fallback_plans.sort(key=lambda x: (abs(x["beta"]), -x["swap"]))
                final_best = fallback_plans[0]
                is_fallback = True
            
            if final_best is None:
                st.error(f"❌ 計算可能な組み合わせが1つも見つかりませんでした。")
                if 'results' in st.session_state: del st.session_state['results']
            else:
                st.session_state['results'] = {
                    'best': final_best,
                    'is_fallback': is_fallback,
                    'df_full': df_full,
                    'calc_period': calc_period_option,
                    'target_notional': target_notional,
                    'capital': capital,
                    'current_rates': current_rates
                }

# --- 結果表示 ---
if 'results' in st.session_state:
    res = st.session_state['results']
    best = res['best']
    is_fallback = res.get('is_fallback', False)
    df_full = res['df_full']
    target_notional = res['target_notional']
    calc_capital = res['capital']
    current_rates = res['current_rates']
    
    best_swap_val = best['swap'] if not np.isnan(best['swap']) else 0

    if is_fallback:
        st.warning("⚠️ 条件（β・相関）を完全に満たすプランが見つかりませんでした。")
        st.markdown(f"**参考として、条件外の中で最もβが低く安全なプランを表示します。**")
    else:
        st.success("🎉 計算完了！最適なプランが見つかりました")
    
    st.info(f"最適化基準: {res['calc_period']} のデータを使用")

    m1, m2, m3 = st.columns(3)
    m1.metric("💰 予想日次スワップ", f"¥{int(best_swap_val):,}")
    m1.metric("📈 予想年利", f"{(best_swap_val * 365 / calc_capital * 100):.1f}%")
    m2.metric("⚖️ ポートフォリオβ", f"{best['beta']:.4f}")
    m3.metric("🛡️ 必要証拠金 (目安)", f"¥{int(target_notional / 25):,}")

    st.subheader("📋 注文レシピ")
    orders = []
    side_notional = target_notional / 2
    for ccy, w in best['buy'].items():
        rate = current_rates.get(ccy, 0)
        if rate > 0:
            lots = (side_notional * w) / (rate * DEFAULT_LOT_SIZE[ccy])
            orders.append({"売買": "買い", "通貨ペア": ccy, "比率": f"{w*100:.0f}%", "推奨ロット": round(lots, 2)})
    for ccy, w in best['sell'].items():
        rate = current_rates.get(ccy, 0)
        if rate > 0:
            lots = (side_notional * w) / (rate * DEFAULT_LOT_SIZE[ccy])
            orders.append({"売買": "売り", "通貨ペア": ccy, "比率": f"{w*100:.0f}%", "推奨ロット": round(lots, 2)})
    st.dataframe(pd.DataFrame(orders), hide_index=True)

    st.markdown("---")
    
    st.subheader(f"📊 バックテスト ({plot_period_option})")
    
    if "1年" in plot_period_option:
        df_plot = df_full.tail(250)
    elif "2年" in plot_period_option:
        df_plot = df_full.tail(500)
    else:
        df_plot = df_full
    
    buy_series = pd.Series(0.0, index=df_plot.index)
    for ccy, w in best['buy'].items(): buy_series += df_plot[ccy] * w
    
    sell_series = pd.Series(0.0, index=df_plot.index)
    for ccy, w in best['sell'].items(): sell_series += df_plot[ccy] * w
    
    daily_capital_pl = (buy_series - sell_series) * side_notional
    total_pl = (daily_capital_pl + best_swap_val).cumsum()
    capital_only = daily_capital_pl.cumsum()
    
    total_pl_pct = (total_pl / calc_capital) * 100
    capital_only_pct = (capital_only / calc_capital) * 100
    
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=total_pl.index, y=total_pl_pct.values, name='合計損益 (%)', line=dict(color='green', width=2)))
    fig_bt.add_trace(go.Scatter(x=capital_only.index, y=capital_only_pct.values, name='為替損益のみ (%)', line=dict(color='gray', dash='dot')))
    
    fig_bt.update_layout(
        title=f"損益推移 (対元本比率)", 
        height=400,
        yaxis_title="損益率 (%)",
        yaxis_ticksuffix="%"
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    buy_nav = (1 + buy_series).cumprod() * 100
    sell_nav = (1 + sell_series).cumprod() * 100
    
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(x=buy_nav.index, y=buy_nav.values, name="買いバスケット", line=dict(color='blue')))
    fig_corr.add_trace(go.Scatter(x=sell_nav.index, y=sell_nav.values, name="売りバスケット", line=dict(color='red')))
    fig_corr.update_layout(title="動きの比較 (相関)", height=400)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.info(f"💡 **最適化期間({res['calc_period']})での相関係数: {best['corr']:.4f}**")
