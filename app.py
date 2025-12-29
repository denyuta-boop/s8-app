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
def fetch_data(days=730):
    """データ取得 (デフォルト2年分)"""
    try:
        symbols = list(TICKER_MAP.values())
        data = yf.download(symbols, period=f"{days}d", progress=False, auto_adjust=False)
        
        if data.empty: return None, {}, None
        
        df_close = pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            try:
                if 'Close' in data.columns.get_level_values(0):
                    df_close = data['Close'].copy()
                elif 'Adj Close' in data.columns.get_level_values(0):
                    df_close = data['Adj Close'].copy()
                else:
                    df_close = data.copy()
                    df_close.columns = df_close.columns.droplevel(0)
            except:
                df_close = data.copy()
        else:
            if 'Close' in data.columns:
                 df_close = data[['Close']].copy()
            else:
                 df_close = data.copy()

        final_df = pd.DataFrame(index=df_close.index)
        for col in df_close.columns:
            col_str = str(col).upper()
            matched_name = None
            for internal_name, yahoo_symbol in TICKER_MAP.items():
                search_key = yahoo_symbol.upper().replace("=X", "")
                if search_key in col_str:
                    matched_name = internal_name
                    break
            if matched_name:
                final_df[matched_name] = df_close[col]

        if final_df.empty: return None, {}, None
        final_df = final_df.dropna(axis=1, how='all')
        df_filled = final_df.ffill().bfill()
        df_filled = df_filled.dropna(how='all')

        if len(df_filled) < 10: return None, {}, None

        latest_rates = df_filled.iloc[-1].to_dict()
        returns = np.log(df_filled).diff().dropna()
        
        return returns, latest_rates, df_filled
    except Exception as e:
        return None, {}, None

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
    """精密モード: 10%刻みの重み生成"""
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
    
    password = st.text_input("🔑 パスワード", type="password")
    if password != "s6secret":
        st.warning("パスワードを入力してください")
        st.stop()

    capital = st.number_input("💰 運用資金 (円)", value=1000000, step=100000)
    leverage = st.number_input("⚙️ 目標レバレッジ (倍)", value=16.0, step=0.1)

    st.subheader("🛡️ リスク制御")
    target_beta = st.slider("許容するβの範囲 (±)", 0.01, 0.20, 0.05, step=0.01, help="推奨: 0.05以下")
    target_corr = st.slider("最低相関係数", 0.0, 1.0, 0.80, step=0.05, help="買いと売りの動きの一致度。推奨: 0.8以上")
    
    st.markdown("---")
    st.caption("通貨保有比率の制限")
    
    other_limit = st.slider("🌍 TRY以外の最大比率制限 (%)", 30, 100, 70, step=10, help="特定の通貨への集中を防ぎます。")
    try_limit = st.slider("🇹🇷 TRYJPYの最大比率制限 (%)", 0, 100, 100, step=10)
    
    st.subheader("🔢 構成通貨数")
    buy_count_range = st.slider("買い通貨ペア数 (範囲)", 1, 4, (2, 4))
    sell_count_range = st.slider("売り通貨ペア数 (範囲)", 1, 4, (2, 3))

    with st.expander("📝 スワップポイント設定", expanded=False):
        swap_inputs = {}
        for ccy, val in DEFAULT_SWAP.items():
            swap_inputs[ccy] = st.number_input(f"{ccy}", value=float(val), step=0.1)

# --- メイン画面 ---
st.title("📱 S6戦略 自動最適化ツール")

col1, col2 = st.columns(2)
with col1:
    buy_candidates = st.multiselect("📈 買い候補", 
                                    ["MXNJPY", "ZARJPY", "PLNJPY", "TRYJPY", "CZKJPY"],
                                    default=["MXNJPY", "ZARJPY", "PLNJPY", "TRYJPY", "CZKJPY"])
with col2:
    sell_candidates = st.multiselect("📉 売り候補", 
                                     ["USDJPY", "CHFJPY", "EURJPY"],
                                     default=["USDJPY", "CHFJPY", "EURJPY"])

if st.button("🚀 計算スタート", type="primary"):
    
    if len(buy_candidates) < buy_count_range[0] or len(sell_candidates) < sell_count_range[0]:
        st.error("⚠️ エラー: 候補通貨の数が、指定された最小構成数より少ないです。")
        st.stop()

    with st.spinner("⏳ データ取得＆最適化計算中..."):
        # グラフ表示に合わせて2年分(730日)を取得
        df_returns, current_rates, df_prices = fetch_data(days=730)
        
        if df_returns is None or df_returns.empty:
            st.error("❌ データ取得エラー。")
            st.stop()
            
        betas = {}
        if "USDJPY" not in df_returns.columns:
            st.error(f"❌ USDJPYデータ不足 (取得列: {list(df_returns.columns)})")
            st.stop()
            
        for col in df_returns.columns:
            if col == "USDJPY": betas[col] = 1.0
            else: betas[col] = calculate_beta(df_returns[col], df_returns["USDJPY"])
            
        target_notional = capital * leverage
        valid_plans = []

        # --- 高速化のための事前計算 ---
        
        # 1. 買いパターンの生成と事前計算
        buy_precalc = []
        for size in range(buy_count_range[0], min(buy_count_range[1], len(buy_candidates)) + 1):
            for combo in itertools.combinations(buy_candidates, size):
                if not all(ccy in betas for ccy in combo): continue

                weights_list = generate_weights(size)
                for wp in weights_list:
                    pattern = {combo[i]: wp[i] for i in range(size)}
                    
                    # 保有比率制限
                    is_valid_weight = True
                    for ccy, weight in pattern.items():
                        if ccy == "TRYJPY":
                            if weight > (try_limit / 100): is_valid_weight = False; break
                        else:
                            if weight > (other_limit / 100): is_valid_weight = False; break
                    
                    if not is_valid_weight: continue

                    b_beta = sum(betas.get(ccy, 0) * w for ccy, w in pattern.items())
                    
                    b_series = pd.Series(0.0, index=df_returns.index)
                    for ccy, w in pattern.items():
                         b_series += df_returns[ccy] * w
                    
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
                            "pattern": pattern,
                            "beta": b_beta,
                            "series": b_series,
                            "swap": daily_swap_buy
                        })

        # 2. 売りパターンの生成と事前計算
        sell_precalc = []
        for size in range(sell_count_range[0], min(sell_count_range[1], len(sell_candidates)) + 1):
            for combo in itertools.combinations(sell_candidates, size):
                if not all(ccy in betas for ccy in combo): continue

                weights_list = generate_weights(size)
                for wp in weights_list:
                    pattern = {combo[i]: wp[i] for i in range(size)}
                    
                    s_beta = sum(betas.get(ccy, 0) * w for ccy, w in pattern.items()) * -1
                    s_series = pd.Series(0.0, index=df_returns.index)
                    for ccy, w in pattern.items():
                         s_series += df_returns[ccy] * w
                    
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
                            "pattern": pattern,
                            "beta": s_beta,
                            "series": s_series,
                            "swap": daily_swap_sell
                        })

        # 3. 総当たりマッチング
        for b_item in buy_precalc:
            for s_item in sell_precalc:
                
                net_beta = b_item["beta"] + s_item["beta"]
                if abs(net_beta) >= target_beta: continue
                
                corr = b_item["series"].corr(s_item["series"])
                if np.isnan(corr): corr = 0
                if corr < target_corr: continue
                
                total_swap = b_item["swap"] + s_item["swap"]
                
                valid_plans.append({
                    "buy": b_item["pattern"],
                    "sell": s_item["pattern"],
                    "beta": net_beta,
                    "swap": total_swap,
                    "corr": corr
                })

        if not valid_plans:
            st.error(f"❌ 条件に合うプランが見つかりませんでした。\n・β < {target_beta}\n・相関 > {target_corr}\n・通貨比率制限あり\n\n条件を緩めるか、候補を増やしてください。")
        else:
            valid_plans.sort(key=lambda x: x["swap"], reverse=True)
            best = valid_plans[0]
            
            best_swap_val = best['swap']
            if np.isnan(best_swap_val): best_swap_val = 0

            st.success("🎉 計算完了！最適なプランが見つかりました")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("💰 予想日次スワップ", f"¥{int(best_swap_val):,}")
            m1.metric("📈 予想年利", f"{(best_swap_val * 365 / capital * 100):.1f}%")
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
            
            # グラフ描画
            buy_series = pd.Series(0.0, index=df_returns.index)
            for ccy, w in best['buy'].items():
                 buy_series += df_returns[ccy] * w
            
            sell_series = pd.Series(0.0, index=df_returns.index)
            for ccy, w in best['sell'].items():
                 sell_series += df_returns[ccy] * w
            
            daily_capital_pl = (buy_series - sell_series) * side_notional
            total_pl = (daily_capital_pl + best_swap_val).cumsum()
            capital_only = daily_capital_pl.cumsum()
            
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=total_pl.index, y=total_pl.values, name='合計損益', line=dict(color='green', width=2)))
            fig_bt.add_trace(go.Scatter(x=capital_only.index, y=capital_only.values, name='為替損益のみ', line=dict(color='gray', dash='dot')))
            # ★ここを修正しました
            fig_bt.update_layout(title="📈 過去2年間の損益シミュレーション (バックテスト)", height=400)
            st.plotly_chart(fig_bt, use_container_width=True)

            buy_nav = (1 + buy_series).cumprod() * 100
            sell_nav = (1 + sell_series).cumprod() * 100
            
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(x=buy_nav.index, y=buy_nav.values, name="買いバスケット", line=dict(color='blue')))
            fig_corr.add_trace(go.Scatter(x=sell_nav.index, y=sell_nav.values, name="売りバスケット", line=dict(color='red')))
            fig_corr.update_layout(title="🤝 相関チェック (動きが同じならOK)", height=400)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.info(f"💡 **相関係数: {best['corr']:.4f}** (1.0に近いほどリスクヘッジが効いています)")
