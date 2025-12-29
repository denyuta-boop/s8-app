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
def fetch_data(days=365):
    """データ取得 (超・堅牢版)"""
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
    """精密モード: 10%刻みの重み生成 (2〜4通貨対応)"""
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
    # ★追加: 4通貨分散パターン
    elif n == 4:
        # 計算量削減のため20%刻みも検討するが、一旦10%刻みで実装(Renderなら耐えるはず)
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
    try_limit = st.slider("🇹🇷 TRYJPYの最大比率制限 (%)", 0, 100, 100, step=10)
    
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
    
    if len(buy_candidates) < 2 or len(sell_candidates) < 1:
        st.error("⚠️ エラー: 買い候補は2つ以上、売り候補は1つ以上選んでください。")
        st.stop()

    with st.spinner("⏳ データ取得＆最適化計算中... (パターン数が多いと少し時間がかかります)"):
        df_returns, current_rates, df_prices = fetch_data(days=730)
        
        if df_returns is None or df_returns.empty:
            st.error("❌ データ取得エラー。Yahoo Financeからデータを取得できませんでした。")
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
        
        # --- 組み合わせ生成ロジックの変更 ---
        # 2通貨ペア, 3通貨ペア, 4通貨ペア をすべて試し、一番良いものを探す
        buy_patterns_all = []
        
        # 探索するサイズ: 2〜4 (ただし候補数が足りない場合はそこまで)
        max_size = min(4, len(buy_candidates))
        
        for size in range(2, max_size + 1):
            for combo in itertools.combinations(buy_candidates, size):
                weights_list = generate_weights(size)
                for wp in weights_list:
                    # wpは {0: 0.1, 1: 0.9} のようなインデックスキーなので、通貨名キーに変換
                    pattern = {combo[i]: wp[i] for i in range(size)}
                    buy_patterns_all.append(pattern)

        # 売りは最大2通貨分散まで（複雑化しすぎるため）
        sell_patterns_all = []
        sell_max_size = min(2, len(sell_candidates))
        for size in range(1, sell_max_size + 1):
            for combo in itertools.combinations(sell_candidates, size):
                if size == 1:
                    sell_patterns_all.append({combo[0]: 1.0})
                else:
                    weights_list = generate_weights(size)
                    for wp in weights_list:
                        pattern = {combo[i]: wp[i] for i in range(size)}
                        sell_patterns_all.append(pattern)
        
        # --- 総当たり計算 ---
        for b_pat in buy_patterns_all:
            # データ確認
            if not all(ccy in betas for ccy in b_pat): continue

            # TRY制限チェック
            if "TRYJPY" in b_pat:
                if b_pat["TRYJPY"] > (try_limit / 100): continue

            b_beta = sum(betas.get(ccy, 0) * w for ccy, w in b_pat.items())
            
            for s_pat in sell_patterns_all:
                if not all(ccy in betas for ccy in s_pat): continue
                
                s_beta = sum(betas.get(ccy, 0) * w for ccy, w in s_pat.items()) * -1
                net_beta = b_beta + s_beta
                
                if abs(net_beta) < target_beta:
                    side_notional = target_notional / 2
                    daily_swap = 0
                    try:
                        for ccy, w in b_pat.items():
                            rate = current_rates.get(ccy, 0)
                            if rate == 0: continue
                            lots = (side_notional * w) / (rate * DEFAULT_LOT_SIZE[ccy])
                            daily_swap += lots * swap_inputs.get(ccy, 0)
                        for ccy, w in s_pat.items():
                            rate = current_rates.get(ccy, 0)
                            if rate == 0: continue
                            lots = (side_notional * w) / (rate * DEFAULT_LOT_SIZE[ccy])
                            daily_swap += lots * swap_inputs.get(ccy, 0)
                        
                        if np.isnan(daily_swap) or daily_swap == 0: continue
                        valid_plans.append({"buy": b_pat, "sell": s_pat, "beta": net_beta, "swap": daily_swap})
                    except: continue

        if not valid_plans:
            st.error(f"❌ 条件(β < {target_beta})に合う組み合わせが見つかりませんでした。条件を緩めるか、候補を増やしてください。")
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
            
            # グラフ用データ
            buy_series = pd.Series(0.0, index=df_returns.index)
            valid_buy = True
            for ccy, w in best['buy'].items():
                if ccy in df_returns.columns: 
                    buy_series += df_returns[ccy] * w
                else: valid_buy = False
            
            sell_series = pd.Series(0.0, index=df_returns.index)
            valid_sell = True
            for ccy, w in best['sell'].items():
                if ccy in df_returns.columns: 
                    sell_series += df_returns[ccy] * w
                else: valid_sell = False
            
            if valid_buy and valid_sell:
                daily_capital_pl = (buy_series - sell_series) * side_notional
                total_pl = (daily_capital_pl + best_swap_val).cumsum()
                capital_only = daily_capital_pl.cumsum()
                
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=total_pl.index, y=total_pl.values, name='合計損益', line=dict(color='green', width=2)))
                fig_bt.add_trace(go.Scatter(x=capital_only.index, y=capital_only.values, name='為替損益のみ', line=dict(color='gray', dash='dot')))
                fig_bt.update_layout(title="📈 1年間の損益シミュレーション", height=400)
                st.plotly_chart(fig_bt, use_container_width=True)

                buy_nav = (1 + buy_series).cumprod() * 100
                sell_nav = (1 + sell_series).cumprod() * 100
                
                fig_corr = go.Figure()
                fig_corr.add_trace(go.Scatter(x=buy_nav.index, y=buy_nav.values, name="買いバスケット", line=dict(color='blue')))
                fig_corr.add_trace(go.Scatter(x=sell_nav.index, y=sell_nav.values, name="売りバスケット", line=dict(color='red')))
                fig_corr.update_layout(title="🤝 相関チェック (動きが同じならOK)", height=400)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                corr = buy_series.corr(sell_series)
                if np.isnan(corr): corr = 0.0
                st.info(f"💡 **相関係数: {corr:.4f}** (1.0に近いほどリスクヘッジが効いています)")
            else:
                st.warning("⚠️ データの履歴不足により、グラフを描画できませんでした。")
