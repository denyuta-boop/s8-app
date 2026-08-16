# CLAUDE.md

## アプリ概要

`app.py` 1ファイル構成の Streamlit アプリ。FX キャリートレード戦略の最適なポートフォリオ（買い通貨バスケット × 売り通貨バスケット）を総当たりで探索し、スワップ収益・シャープレシオ・カルマーレシオによってスコアリングして最良プランを選出する。

## 主要関数

| 関数 | 役割 |
|------|------|
| `fetch_data(days=1095)` | yfinance から為替データを取得しlog対数リターンを返す。`@st.cache_data(ttl=3600)` でキャッシュ。HUFJPY は下記の通り合成。 |
| `calculate_beta(asset, benchmark)` | 線形回帰で対USDJPYのβを計算 |
| `generate_weights(n)` | n通貨の重みの組み合わせを10%刻みで列挙 |
| `calc_sharpe(...)` | 日次損益をcapitalで正規化し年率シャープレシオを算出 |
| `calc_calmar(...)` | 累積損益の最大ドローダウンと年率リターンの比。`np.clip(-10, 10)` で爆発防止。 |

## スコア計算

最適化モードによってスコア式が異なる：

```
シャープレシオ最大: score = sharpe + calmar_weight * calmar
スワップ最大:       score = (total_swap * 365 / capital) + calmar_weight * calmar
カスタム加重:       score = (sw*norm_swap + sh*norm_sharpe + cal*norm_calmar) / (sw+sh+cal)
```

- `calmar` は `[-10, 10]` にクリップ済み
- カスタム加重はmin-max正規化後に3ウェイトの合計で除算（スコア上限=1.0）

## データフロー

```
fetch_data() → df_full (log returns, 全期間)
             → df_calc (tail(calc_days), 最適化に使用)
             → df_prices (生価格, 内部でHUFJPY合成のみに使用)

最適化ループ: buy_precalc × sell_precalc の総当たり
  → beta フィルタ → 相関フィルタ → スコア計算
  → valid_plans (相関OK) / fallback_plans (相関NG)
  → 上位1件を session_state に保存

結果表示: session_state['results'] から読み出し
```

## HUFJPY の合成

Yahoo Finance に `HUFJPY=X` が存在しないため、`fetch_data()` 内で以下の式で合成している：

```
HUFJPY = USDJPY / USDHUF
```

「1ドル = X円」÷「1ドル = Yフォリント」=「1フォリント = X/Y 円」。
合成後は `USDHUF` 列を削除し、他の通貨ペアと同じように扱う。
yfinance 側で USDHUF のデータが取れない場合は HUFJPY も存在しない列になるため、最適化ループ前の欠損チェックで検出される。

## 認証

パスワードは `st.secrets["APP_PASSWORD"]` → 環境変数 `APP_PASSWORD` の順で取得。どちらも未設定の場合は常にデモモード（通貨2種のみ）。ソースコードにパスワードを書かないこと。

## 開発上の注意

- `df_full` という変数名だが実体は**log returns**（価格ではない）。混同注意。
- `calc_calmar` の docstring に `/ capital` と書かれているが実装では省略（分子・分母で相殺されるため比率は同値）。
- `fetch_data` は `@st.cache_data` でキャッシュされるため、データ変更を確認したい場合はブラウザ側で「Clear cache」が必要。
- 計算ボタン押下後の結果は `st.session_state['results']` に格納され、サイドバー設定を変えても再計算されるまで表示は更新されない。
