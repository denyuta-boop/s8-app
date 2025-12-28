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

# 初期スワップ値
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
def fetch_data(tickers, days=365):
    """データ取得 (エラー回避の強化版)"""
    try:
        # データを取得
        data = yf.download(tickers, period=f"{days}d", progress=False, auto_adjust=True)
        if data.empty: return None, {}, None
        
        # カラム構造の正規化
        if isinstance(data.columns, pd.MultiIndex):
            try: df = data["Close"]
            except KeyError: df = data
        else:
            df = data["Close"] if "Close" in data.columns else data

        # データの穴埋め
        df_filled = df.ffill().bfill()
        
        # 最新レート取得
        latest_rates = df_filled.iloc[-1].to_dict()
        
        # リターン計算
        returns = np.log(df_filled).diff().dropna()
        
        return returns, latest_rates, df_filled
    except Exception as e:
        return None, {}, None

def calculate_beta(asset_returns, benchmark_returns):
    idx = asset_returns.index.intersection(benchmark_returns.index)
    if len(idx) < 10: return 0
    slope, _, _, _, _ = stats.linregress(benchmark_returns.loc[idx], asset_returns.loc[idx])
    return slope

def generate_weights(n):
    """精密モード: 10%刻みの重み生成"""
    weights = []
    if n == 1: return [{0: 1.0}]
    elif n == 2:
        for i in range(1, 10): weights.append({0: i/10, 1: (10-i)/10})
    elif n == 3:
        for i in range(1, 9):
            for j in range(1, 9-i):
                k = 10 - i - j
                if k > 0: weights.append({0: i/10, 1: j/10, 2: k/10})
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

    with st.spinner("⏳ データ取得＆最適化計算中..."):
        # 1. データ取得
        all_tickers = list(set(buy_candidates + sell_candidates + ["USDJPY"]))
        yf_tickers = [TICKER_MAP[t] for t in all_tickers]
        
        df_returns, latest_rates_raw, df_prices = fetch_data(yf_tickers)
        
        if df_returns is None:
            st.error("❌ データ取得エラー。Yahoo Financeに接続できませんでした。")
            st.stop()
            
        # ★ここが修正ポイント: カラム名の強制クリーニング
        # MXNJPY=X が来ても mxnjpy=x が来ても MXNJPY に統一する
        inv_map = {v: k for k, v in TICKER_MAP.items()}
        
        new_cols = []
        for c in df_returns.columns:
            # 文字列化して =X を削除し、大文字に統一
            clean_name = str(c).upper().replace("=X", "").replace("=x", "")
            # TICKER_MAPのキーにあるか確認
            if clean_name in TICKER_MAP:
                new_cols.append(clean_name)
            else:
                # 見つからない場合は元のカラム名を使用（マッピング試行）
                new_cols.append(inv_map.get(c, c))
        
        df_returns.columns = new_cols
        
        # レート辞書の整理
        current_rates = {}
        for k, v in latest_rates_raw.items():
            # キーがタプルの場合などの処理
            key_str = str(k[1] if isinstance(k, tuple) else k).upper().replace("=X", "")
            
            # マッチング
            if key_str in TICKER_MAP:
                current_rates[key_str] = v
            else:
                # 予備検索
                for t_name, t_code in TICKER_MAP.items():
                    if t_code == k or t_name == k:
                        current_rates[t_name] = v
                        break
        
        # 2. β計算
        betas = {}
        if "USDJPY" not in df_returns.columns:
            st.error(f"❌ USDJPYデータ不足 (取得カラム: {list(df_returns.columns)})")
            st.stop()
            
        for col in df_returns.columns:
            if col == "USDJPY": betas[col] = 1.0
            else: betas[col] = calculate_beta(df_returns[col], df_returns["USDJPY"])
            
        # 3. 総当たりシミュレーション
        target_notional = capital * leverage
        valid_plans = []
        
        buy_combos = []
        if len(buy_candidates) >= 3:
            for combo in itertools.combinations(buy_candidates, 3):
                for wp in generate_weights(3): buy_combos.append({combo[i]: wp[i] for i in range(3)})
        elif len(buy_candidates) >= 2:
            for combo in itertools.combinations(buy_candidates, 2):
                for wp in generate_weights(2): buy_combos.append({combo[i]: wp[i] for i in range(2)})

        sell_combos = []
        if len(sell_candidates) >= 2:
            for combo in itertools.combinations(sell_candidates, 2):
                for wp in generate_weights(2): sell_combos.append({combo[i]: wp[i] for i in range(2)})
        for c in sell_candidates: sell_combos.append({c: 1.0})
        
        for b_pat in buy_combos:
            b_beta = sum(betas.get(ccy, 0) * w for ccy, w in b_pat.items())
            for s_pat in sell_combos:
                s_beta = sum(betas.get(ccy, 0) * w for ccy, w in s_pat.items()) * -1
                net_beta = b_beta + s_beta
                
                if abs(net_beta) < 0.15:
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
            st.error("❌ 条件に合う組み合わせが見つかりませんでした。")
        else:
            # 4. 結果表示
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

            # 5. グラフ描画
            st.markdown("---")
            
            buy_series = pd.Series(0.0, index=df_returns.index)
            for ccy, w in best['buy'].items():
                if ccy in df_returns.columns: buy_series += df_returns[ccy] * w
            sell_series = pd.Series(0.0, index=df_returns.index)
            for ccy, w in best['sell'].items():
                if ccy in df_returns.columns: sell_series += df_returns[ccy] * w
            
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
            if np.isnan(corr): corr = 0.0 # NaN対策
            st.info(f"💡 **相関係数: {corr:.4f}** (1.0に近いほどリスクヘッジが効いています)")
