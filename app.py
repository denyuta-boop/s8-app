import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import plotly.graph_objects as go
import itertools
import time

# --- ページ設定 ---
st.set_page_config(page_title="S8戦略 自動最適化ツール", layout="wide")

# --- 定数 ---
TICKER_MAP = {
    "USDJPY": "USDJPY=X", "MXNJPY": "MXNJPY=X", "PLNJPY": "PLNJPY=X",
    "CZKJPY": "CZKJPY=X", "CHFJPY": "CHFJPY=X", "ZARJPY": "ZARJPY=X",
    "TRYJPY": "TRYJPY=X", "EURJPY": "EURJPY=X",
    "USDHUF": "USDHUF=X",  # HUF追加: 直接レートがないためUSDHUFを取得しHUFJPYを合成する
}

BUY_GROUP = ["MXNJPY", "ZARJPY", "PLNJPY", "TRYJPY", "CZKJPY", "HUFJPY"]  # HUF追加
SELL_GROUP = ["USDJPY", "CHFJPY", "EURJPY"]

DEFAULT_SWAP = {
    "MXNJPY": 14.1, "PLNJPY": 32.0, "ZARJPY": 13.1, "TRYJPY": 25.1,
    "CZKJPY": 5.5, "HUFJPY": 6.0,
    "USDJPY": -155.0, "CHFJPY": 35.0, "EURJPY": -70.0
}

DEFAULT_LOT_UNIT = {
    "MXNJPY": 10000, "ZARJPY": 10000, "PLNJPY": 10000,
    "TRYJPY": 10000, "CZKJPY": 10000, "HUFJPY": 100000,  # HUFは100,000通貨単位
    "USDJPY": 10000, "CHFJPY": 10000, "EURJPY": 10000,
}

# --- 関数定義 ---
@st.cache_data(ttl=3600)
def fetch_data(days=1095):
    debug_logs = []
    try:
        usd_symbol = "USDJPY=X"
        other_symbols = [v for k, v in TICKER_MAP.items() if v != usd_symbol]

        df_usd_clean = pd.DataFrame()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data_usd = yf.download(usd_symbol, period=f"{days}d", progress=False, auto_adjust=False)
                if not data_usd.empty:
                    target_col = (
                        data_usd['Close'] if 'Close' in data_usd.columns
                        else data_usd['Adj Close'] if 'Adj Close' in data_usd.columns
                        else data_usd.iloc[:, 0]
                    )
                    df_usd_clean["USDJPY"] = target_col
                    break
            except Exception as e:
                debug_logs.append(f"USDJPY Attempt {attempt+1} failed: {str(e)}")
            time.sleep(1)

        if df_usd_clean.empty:
            return None, {}, None, debug_logs

        df_others_clean = pd.DataFrame()
        for attempt in range(max_retries):
            try:
                data_others = yf.download(other_symbols, period=f"{days}d", progress=False, auto_adjust=False)
                if not data_others.empty:
                    if isinstance(data_others.columns, pd.MultiIndex):
                        df_temp = (
                            data_others['Close'] if 'Close' in data_others.columns.get_level_values(0)
                            else data_others['Adj Close']
                        )
                    else:
                        df_temp = data_others['Close'] if 'Close' in data_others.columns else data_others
                    for col in df_temp.columns:
                        col_str = str(col).upper()
                        matched_name = next(
                            (k for k, v in TICKER_MAP.items() if k != "USDJPY" and v.replace("=X", "") in col_str),
                            None
                        )
                        if matched_name:
                            df_others_clean[matched_name] = df_temp[col]
                    break
            except Exception as e:
                # FIX: 空のexceptを修正。エラー内容をデバッグログに記録する
                debug_logs.append(f"Others Attempt {attempt+1} failed: {str(e)}")
                time.sleep(1)

        final_df = df_usd_clean.join(df_others_clean, how='outer').ffill().bfill()

        # HUF追加: HUFJPY = USDJPY ÷ USDHUF で合成（価格ベースで行い、その後リターンを算出）
        if "USDJPY" in final_df.columns and "USDHUF" in final_df.columns:
            final_df["HUFJPY"] = final_df["USDJPY"] / final_df["USDHUF"]
            final_df.drop(columns=["USDHUF"], inplace=True)
        elif "USDHUF" in final_df.columns:
            final_df.drop(columns=["USDHUF"], inplace=True)  # 合成失敗時も中間列は除去

        if final_df.empty or len(final_df) < 10:
            return None, {}, None, debug_logs

        latest_rates = final_df.iloc[-1].to_dict()
        returns = np.log(final_df).diff().dropna()
        return returns, latest_rates, final_df, debug_logs
    except Exception as e:
        debug_logs.append(f"Fatal: {str(e)}")
        return None, {}, None, debug_logs


def calculate_beta(asset_returns, benchmark_returns):
    common_idx = asset_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 10:
        return 0.0
    y, x = asset_returns.loc[common_idx], benchmark_returns.loc[common_idx]
    if x.std() == 0 or y.std() == 0:
        return 0.0
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope if not np.isnan(slope) else 0.0


def generate_weights(n):
    """
    通貨数の重みを10%刻みで生成します。
    FIX: 最後の重みを整数演算で確定させることで浮動小数点の累積誤差を防止します。
    例: w5 = 1.0 - w1 - w2 - w3 - w4 (誤差が累積) →
        m = step - i - j - k - l; w5 = m/step (整数ベースで正確)
    """
    step = 10
    weights = []

    if n == 1:
        return [{0: 1.0}]

    elif n == 2:
        for i in range(1, step):
            j = step - i
            weights.append({0: i / step, 1: j / step})

    elif n == 3:
        for i in range(1, step - 1):
            for j in range(1, step - i):
                k = step - i - j
                if k > 0:
                    weights.append({0: i / step, 1: j / step, 2: k / step})

    elif n == 4:
        for i in range(1, step - 2):
            for j in range(1, step - i - 1):
                for k in range(1, step - i - j):
                    l = step - i - j - k
                    if l > 0:
                        weights.append({0: i / step, 1: j / step, 2: k / step, 3: l / step})

    elif n == 5:
        for i in range(1, step - 3):
            for j in range(1, step - i - 2):
                for k in range(1, step - i - j - 1):
                    for l in range(1, step - i - j - k):
                        m = step - i - j - k - l
                        if m > 0:
                            weights.append({0: i / step, 1: j / step, 2: k / step, 3: l / step, 4: m / step})

    return weights


# --- サイドバー ---
with st.sidebar:
    st.header("⚙️ 設定パネル")

    password = st.text_input("🔑 パスワード (未入力でデモモード)", type="password")
    is_demo_mode = password != "s6secret"

    if not is_demo_mode:
        st.success("🔓 フル機能モード")
        default_other_limit = 50
        default_buy_range = (2, 4)
        default_sell_range = (2, 3)
    else:
        st.info("👀 デモモード")
        default_other_limit = 100
        default_buy_range = (1, 2)
        default_sell_range = (1, 1)

    capital = st.number_input("💰 運用資金 (円)", value=1000000, step=100000)
    leverage = st.number_input("⚙️ 目標レバレッジ (倍)", value=16.66, step=0.1)

    with st.expander("📝 スワップ & Lot単位設定"):
        swap_inputs = {}
        lot_inputs = {}
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("##### 🟢 買い (受取)")
            for ccy in BUY_GROUP:
                val = DEFAULT_SWAP.get(ccy, 0.0)
                c1, c2 = st.columns([1.2, 1])
                with c1: swap_inputs[ccy] = st.number_input(f"{ccy} Swap", value=float(val), step=0.1, key=f"swap_{ccy}")
                with c2: lot_inputs[ccy] = st.number_input("単位", value=DEFAULT_LOT_UNIT.get(ccy, 10000), step=1000, key=f"lot_{ccy}")
        with col_s2:
            st.markdown("##### 🔴 売り (支払)")
            for ccy in SELL_GROUP:
                val = DEFAULT_SWAP.get(ccy, 0.0)
                c1, c2 = st.columns([1.2, 1])
                with c1: swap_inputs[ccy] = st.number_input(f"{ccy} Swap", value=float(val), step=0.1, key=f"swap_{ccy}")
                with c2: lot_inputs[ccy] = st.number_input("単位", value=DEFAULT_LOT_UNIT.get(ccy, 10000), step=1000, key=f"lot_{ccy}")

    st.markdown("---")
    st.subheader("🛡️ リスク制御")
    calc_period_option = st.selectbox("β・相関計算期間", ["直近1年 (推奨)", "直近2年", "直近3年"], index=0)
    target_beta = st.slider("許容β (±)", 0.01, 0.50, 0.05, 0.01)
    target_corr = st.slider("最低相関係数", -1.0, 1.0, 0.80, 0.05)

    st.subheader("個別通貨の比率制限（買いのみ）")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**TRYJPY**")
        try_min_pct = st.slider("最低 %", 0, 50, 5, 5, key="try_min")
        try_max_pct = st.slider("最高 %", try_min_pct, 100, 95, 5, key="try_max")
    with col_b:
        st.markdown("**MXNJPY**")
        mxn_min_pct = st.slider("最低 %", 0, 50, 5, 5, key="mxn_min")
        mxn_max_pct = st.slider("最高 %", mxn_min_pct, 100, 95, 5, key="mxn_max")
    with col_c:
        st.markdown("**ZARJPY**")
        zar_min_pct = st.slider("最低 %", 0, 50, 5, 5, key="zar_min")
        zar_max_pct = st.slider("最高 %", zar_min_pct, 100, 95, 5, key="zar_max")

    st.caption("その他の通貨の上限")
    other_limit = st.slider("TRY/MXN/ZAR以外の上限 %", 10, 100, default_other_limit, 10)

    st.subheader("必須通貨の設定（買いのみ）")
    force_include = {}
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1: force_include["TRYJPY"] = st.checkbox("TRYJPY を必ず入れる", False, key="force_try")
    with col_f2: force_include["MXNJPY"] = st.checkbox("MXNJPY を必ず入れる", False, key="force_mxn")
    with col_f3: force_include["ZARJPY"] = st.checkbox("ZARJPY を必ず入れる", False, key="force_zar")

    st.subheader("🔢 構成通貨数")
    buy_count_range = st.slider("買い通貨ペア数", 1, 5, default_buy_range)
    sell_count_range = st.slider("売り通貨ペア数", 1, 4, default_sell_range)

    st.markdown("---")
    st.subheader("📈 グラフ表示")
    plot_period_option = st.radio("バックテスト期間", ["直近1年", "直近2年", "直近3年 (全期間)"], index=0)


# --- メイン ---
st.title("📱 S8戦略 自動最適化ツール")

if is_demo_mode:
    st.warning("デモモード：通貨選択が制限されています")
    buy_options = ["MXNJPY", "TRYJPY"]
    buy_default = ["MXNJPY", "TRYJPY"]
    sell_options = ["USDJPY"]
    sell_default = ["USDJPY"]
else:
    buy_options = BUY_GROUP.copy()
    buy_default = buy_options.copy()
    sell_options = SELL_GROUP.copy()
    sell_default = sell_options.copy()

col1, col2 = st.columns(2)
with col1: buy_candidates = st.multiselect("買い候補", buy_options, buy_default)
with col2: sell_candidates = st.multiselect("売り候補", sell_options, sell_default)

if st.button("🚀 計算スタート", type="primary"):
    if len(buy_candidates) < buy_count_range[0] or len(sell_candidates) < sell_count_range[0]:
        st.error(f"買いは最低{buy_count_range[0]}、売りは最低{sell_count_range[0]}選んでください")
        st.stop()

    with st.spinner("データ取得＆最適化中..."):
        df_full, current_rates, df_prices, debug_logs = fetch_data()

        if df_full is None:
            st.error("データ取得失敗")
            with st.expander("デバッグ"): [st.write(log) for log in debug_logs]
            st.stop()

        calc_days = 250 if "1年" in calc_period_option else 500 if "2年" in calc_period_option else 750
        df_calc = df_full.tail(calc_days)

        if "USDJPY" not in df_calc.columns:
            st.error("USDJPYデータなし")
            st.stop()

        betas = {
            col: 1.0 if col == "USDJPY" else calculate_beta(df_calc[col], df_calc["USDJPY"])
            for col in df_calc.columns
        }

        target_notional = capital * leverage
        valid_plans = []
        fallback_plans = []
        rejected_by_ratio = total_combinations = 0

        forced_count = sum(force_include.values())
        if forced_count > buy_count_range[1]:
            st.error(f"必須通貨が{forced_count}個ありますが、最大構成数は{buy_count_range[1]}です。設定を修正してください。")
            st.stop()

        min_sum = 0.0
        if force_include.get("TRYJPY"): min_sum += try_min_pct / 100
        if force_include.get("MXNJPY"): min_sum += mxn_min_pct / 100
        if force_include.get("ZARJPY"): min_sum += zar_min_pct / 100
        if min_sum > 1.01:
            st.warning("必須通貨の最低比率合計が100%を超えています。条件が厳しすぎる可能性があります。")

        buy_precalc = []
        for size in range(buy_count_range[0], min(buy_count_range[1], len(buy_candidates)) + 1):
            for combo in itertools.combinations(buy_candidates, size):
                combo_set = set(combo)

                if not all(ccy in combo_set for ccy, must in force_include.items() if must):
                    continue

                weights_list = generate_weights(size)
                for wp in weights_list:
                    pattern = {combo[i]: wp[i] for i in range(size)}
                    total_combinations += 1

                    is_valid = True
                    for ccy, w in pattern.items():
                        w_pct = w * 100
                        if ccy == "TRYJPY":
                            if w_pct < try_min_pct or w_pct > try_max_pct: is_valid = False
                        elif ccy == "MXNJPY":
                            if w_pct < mxn_min_pct or w_pct > mxn_max_pct: is_valid = False
                        elif ccy == "ZARJPY":
                            if w_pct < zar_min_pct or w_pct > zar_max_pct: is_valid = False
                        else:
                            if w_pct > other_limit: is_valid = False
                        if not is_valid: break

                    if not is_valid:
                        rejected_by_ratio += 1
                        continue

                    b_beta = sum(betas.get(ccy, 0) * w for ccy, w in pattern.items())
                    b_series = sum(df_calc[ccy] * w for ccy, w in pattern.items())

                    daily_swap = 0.0
                    side_notional = target_notional / 2
                    valid = True
                    for ccy, w in pattern.items():
                        rate = current_rates.get(ccy, 0)
                        if rate <= 0:
                            valid = False
                            break
                        lots = (side_notional * w) / (rate * lot_inputs.get(ccy, 10000))
                        daily_swap += lots * swap_inputs.get(ccy, 0)

                    if valid:
                        buy_precalc.append({"pattern": pattern, "beta": b_beta, "series": b_series, "swap": daily_swap})

        sell_precalc = []
        for size in range(sell_count_range[0], min(sell_count_range[1], len(sell_candidates)) + 1):
            for combo in itertools.combinations(sell_candidates, size):
                weights_list = generate_weights(size)
                for wp in weights_list:
                    pattern = {combo[i]: wp[i] for i in range(size)}
                    s_beta = sum(betas.get(ccy, 0) * w for ccy, w in pattern.items()) * -1
                    s_series = sum(df_calc[ccy] * w for ccy, w in pattern.items())

                    daily_swap = 0.0
                    side_notional = target_notional / 2
                    valid = True
                    for ccy, w in pattern.items():
                        rate = current_rates.get(ccy, 0)
                        if rate <= 0:
                            valid = False
                            break
                        lots = (side_notional * w) / (rate * lot_inputs.get(ccy, 10000))
                        daily_swap += lots * swap_inputs.get(ccy, 0)

                    if valid:
                        sell_precalc.append({"pattern": pattern, "beta": s_beta, "series": s_series, "swap": daily_swap})

        # FIX: β条件を先にチェックし、不合格の組み合わせは相関計算をスキップ（O(n²)のコスト削減）
        # β不合格のfallbackプランは選出後に相関を遅延計算する
        for b in buy_precalc:
            for s in sell_precalc:
                net_beta = b["beta"] + s["beta"]
                total_swap = b["swap"] + s["swap"]

                if abs(net_beta) < target_beta:
                    corr = b["series"].corr(s["series"])
                    plan = {
                        "buy": b["pattern"], "sell": s["pattern"],
                        "beta": net_beta, "swap": total_swap, "corr": corr
                    }
                    if corr > target_corr:
                        valid_plans.append(plan)
                    else:
                        fallback_plans.append(plan)
                else:
                    # β不合格: 相関計算をスキップし、Seriesを保持して後で補完
                    fallback_plans.append({
                        "buy": b["pattern"], "sell": s["pattern"],
                        "beta": net_beta, "swap": total_swap, "corr": None,
                        "_b_series": b["series"], "_s_series": s["series"]
                    })

        final_best = None
        is_fallback = False
        if valid_plans:
            valid_plans.sort(key=lambda x: x["swap"], reverse=True)
            final_best = valid_plans[0]
        elif fallback_plans:
            fallback_plans.sort(key=lambda x: (abs(x["beta"]), -x["swap"]))
            final_best = fallback_plans[0]
            # β不合格で相関が未計算の場合のみ補完
            if final_best.get("corr") is None:
                final_best["corr"] = final_best["_b_series"].corr(final_best["_s_series"])
            is_fallback = True

        # FIX: session_stateに大きなSeriesオブジェクトを保存しないよう除去
        if final_best:
            final_best.pop("_b_series", None)
            final_best.pop("_s_series", None)

        if final_best is None:
            st.error("有効な組み合わせが見つかりませんでした")
            if rejected_by_ratio == total_combinations and total_combinations > 0:
                st.warning("比率制限が厳しすぎる可能性があります。条件を緩めてください。")
            st.stop()

        st.session_state['results'] = {
            'best': final_best, 'is_fallback': is_fallback,
            'df_full': df_full, 'calc_period': calc_period_option,
            'target_notional': target_notional, 'capital': capital,
            'current_rates': current_rates, 'lot_inputs': lot_inputs,
            'df_calc': df_calc
        }


# --- 結果表示 ---
if 'results' in st.session_state:
    res = st.session_state['results']
    best = res['best']
    is_fallback = res.get('is_fallback', False)
    # FIX: df_calcをsession_stateから取得（ボタン未押下時のNameErrorを修正）
    df_calc = res['df_calc']

    st.subheader("採用通貨の年率標準偏差（最適プラン内）")
    adopted_risks = []
    for ccy, weight in {**best['buy'], **best['sell']}.items():
        if ccy in df_calc.columns:
            daily_std = df_calc[ccy].std()
            annual_std = daily_std * np.sqrt(252) * 100 if daily_std > 0 else 0
            adopted_risks.append({
                "通貨": ccy,
                "比率": f"{weight*100:.0f}%",
                "年率標準偏差": f"{annual_std:.2f}%"
            })
    adopted_df = pd.DataFrame(adopted_risks)
    st.dataframe(adopted_df.sort_values("年率標準偏差", ascending=False), hide_index=True)

    df_full = res['df_full']
    target_notional = res['target_notional']
    calc_capital = res['capital']
    current_rates = res['current_rates']
    lot_inputs = res.get('lot_inputs', {ccy: DEFAULT_LOT_UNIT.get(ccy, 10000) for ccy in TICKER_MAP.keys()})

    best_swap_val = best['swap'] if not np.isnan(best['swap']) else 0

    if is_fallback:
        st.warning("⚠️ 条件（β・相関）を完全に満たすプランが見つかりませんでした。参考プランを表示します。")
    else:
        st.success("🎉 計算完了！最適なプランが見つかりました")

    st.info(f"最適化基準: {res['calc_period']} のデータを使用")
    m1, m2, m3 = st.columns(3)
    m1.metric("💰 予想日次スワップ", f"¥{int(best_swap_val):,}")
    m1.metric("📈 予想年利", f"{(best_swap_val * 365 / calc_capital * 100):.1f}%")
    m2.metric("⚖️ ポートフォリオβ", f"{best['beta']:.4f}")
    m3.metric("🛡️ 最低必要証拠金 (維持率100%)", f"¥{int(target_notional / 25):,}")

    st.subheader("📋 注文レシピ")
    orders = []
    side_notional = target_notional / 2
    for ccy, w in best['buy'].items():
        rate = current_rates.get(ccy, 0)
        if rate > 0:
            amount_jpy = side_notional * w
            unit_size = lot_inputs.get(ccy, DEFAULT_LOT_UNIT.get(ccy, 10000))
            lots = amount_jpy / (rate * unit_size)
            orders.append({
                "売買": "買い", "通貨ペア": ccy,
                "比率": f"{w*100:.0f}%",
                "金額(円)": f"¥{int(amount_jpy):,}",
                "推奨ロット": round(lots, 2)
            })
    for ccy, w in best['sell'].items():
        rate = current_rates.get(ccy, 0)
        if rate > 0:
            amount_jpy = side_notional * w
            unit_size = lot_inputs.get(ccy, DEFAULT_LOT_UNIT.get(ccy, 10000))
            lots = amount_jpy / (rate * unit_size)
            orders.append({
                "売買": "売り", "通貨ペア": ccy,
                "比率": f"{w*100:.0f}%",
                "金額(円)": f"¥{int(amount_jpy):,}",
                "推奨ロット": round(lots, 2)
            })
    st.dataframe(pd.DataFrame(orders), hide_index=True)

    st.markdown("---")
    st.subheader(f"📊 バックテスト ({plot_period_option})")

    if "1年" in plot_period_option: df_plot = df_full.tail(250)
    elif "2年" in plot_period_option: df_plot = df_full.tail(500)
    else: df_plot = df_full

    # 対数リターンを加重合成（バックテスト用）
    buy_log_series = pd.Series(0.0, index=df_plot.index)
    for ccy, w in best['buy'].items(): buy_log_series += df_plot[ccy] * w
    sell_log_series = pd.Series(0.0, index=df_plot.index)
    for ccy, w in best['sell'].items(): sell_log_series += df_plot[ccy] * w

    # FIX: 対数リターン→単純リターンに変換してP&L計算（長期で乖離しないよう精度向上）
    buy_series = np.expm1(buy_log_series)
    sell_series = np.expm1(sell_log_series)

    total_pl = ((buy_series - sell_series) * side_notional + best_swap_val).cumsum()
    capital_only = ((buy_series - sell_series) * side_notional).cumsum()

    total_pl_pct = (total_pl / calc_capital) * 100
    capital_only_pct = (capital_only / calc_capital) * 100

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=total_pl.index, y=total_pl_pct.values,
        name='合計損益 (%)', line=dict(color='green', width=2)
    ))
    fig_bt.add_trace(go.Scatter(
        x=capital_only.index, y=capital_only_pct.values,
        name='為替損益のみ (%)', line=dict(color='gray', dash='dot')
    ))
    fig_bt.update_layout(title="損益推移 (対元本比率)", height=400, yaxis_title="損益率 (%)", yaxis_ticksuffix="%")
    st.plotly_chart(fig_bt, use_container_width=True)

    # FIX: NAVは累積対数リターンから正確に変換（(1+log_r).cumprod()は近似値）
    buy_nav = np.exp(buy_log_series.cumsum()) * 100
    sell_nav = np.exp(sell_log_series.cumsum()) * 100

    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(x=buy_nav.index, y=buy_nav.values, name="買いバスケット", line=dict(color='blue')))
    fig_corr.add_trace(go.Scatter(x=sell_nav.index, y=sell_nav.values, name="売りバスケット", line=dict(color='red')))
    fig_corr.update_layout(title="動きの比較 (相関)", height=400)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("バスケットごとの年率標準偏差（リスク）")
    buy_daily_std = buy_series.std()
    sell_daily_std = sell_series.std()
    portfolio_daily_std = (buy_series - sell_series).std()

    buy_annual_std = buy_daily_std * np.sqrt(252) * 100 if buy_daily_std > 0 else 0
    sell_annual_std = sell_daily_std * np.sqrt(252) * 100 if sell_daily_std > 0 else 0
    portfolio_annual_std = portfolio_daily_std * np.sqrt(252) * 100 if portfolio_daily_std > 0 else 0

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("買いバスケット", f"{buy_annual_std:.2f}%")
    col_r2.metric("売りバスケット", f"{sell_annual_std:.2f}%")
    col_r3.metric("全ポートフォリオ", f"{portfolio_annual_std:.2f}%")

    st.info(
        f"💡 **最適化期間({res['calc_period']})での相関係数: {best['corr']:.4f}** "
        f"（日次の対数変化率で計算。価格水準ではなく値動きの連動性を測っています）"
    )
