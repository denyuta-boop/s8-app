# CLAUDE.md

## このアプリについて

S8戦略（βゼロ構造＋多通貨分散FXキャリートレード）のポートフォリオを自動設計する Streamlit Web アプリ。`app.py` 1ファイル構成。Render でホスティング。

### 戦略の概念（開発背景として理解しておくこと）

- **買いバスケット**：高金利通貨（MXN/ZAR/PLN/TRY/CZK/HUF）を買い → スワップを受け取る
- **売りバスケット**：低金利通貨（USD/CHF/EUR）を売る → 少額スワップを支払うが円の動きをヘッジ
- **βゼロ**：ポートフォリオ全体の「対USDJPY円ベータ」を ±0.05 以内に抑え、円高・円安どちらにも偏らない構造を作る
- **スワップが主役**：為替損益はほぼゼロに抑え、毎日のスワップ収益だけを積み上げる設計
- **分散によるリスク低減**：TRY単体は長期下落リスクが高いが、複数通貨に分散することでボラティリティを個別通貨の約1/3に抑制

### S6 vs S8

S6（初期）はTRYを完全除外した安全版。S8はTRYを10〜20%以内に制限して採用する発展版。旧版はスワップ最大化 = TRY偏重になりがちだったため、シャープレシオ・カルマーレシオによる最適化を追加して**システムが自動的にTRYの入れすぎを抑制**する仕様になった。

---

## ファイル構成

```
app.py            # アプリ全体（サイドバー設定・最適化ループ・結果表示）
requirements.txt  # 依存パッケージ
```

---

## 主要関数

| 関数 | 役割 |
|------|------|
| `fetch_data(days=1095)` | yfinance から為替データを取得し**対数リターン**を返す。`@st.cache_data(ttl=3600)` でキャッシュ済み。戻り値は `(returns_df, latest_rates, price_df, debug_logs)`。 |
| `calculate_beta(asset, benchmark)` | `scipy.stats.linregress` で対USDJPY円ベータを線形回帰計算。 |
| `generate_weights(n)` | n通貨の重みを10%刻みで全列挙。最大n=5まで対応（n=6以上は空リスト）。 |
| `calc_sharpe(...)` | 日次損益をcapitalで正規化して年率シャープレシオを算出。 |
| `calc_calmar(...)` | 累積損益の最大ドローダウンと年率リターンの比。上限を `np.clip(-10, 10)` でキャップ（微小ドローダウンによる爆発防止）。 |

---

## スコア計算

最適化モードによってスコア式が異なる：

```
シャープレシオ最大: score = sharpe + calmar_weight * calmar
スワップ最大:       score = (total_swap * 365 / capital) + calmar_weight * calmar
カスタム加重:       score = (sw*norm_swap + sh*norm_sharpe + cal*norm_calmar) / (sw+sh+cal)
```

- `calmar` は `[-10, 10]` にクリップ済み
- カスタム加重は min-max 正規化後に 3ウェイトの合計で除算（スコア上限 = 1.0）
- スワップ最大モードの `total_swap` は `* 365 / capital` で年率収益率（無次元）に変換してからCalmarと加算。生のJPY値をそのまま加算すると3桁スケール差が生じて Calmar が無意味になるため。

---

## データフロー

```
fetch_data()
  → df_full  : 対数リターン DataFrame（全期間、変数名に注意：価格ではない）
  → df_prices: 生価格 DataFrame（HUFJPY合成後に破棄）
  → latest_rates: 各通貨の最新レート

df_calc = df_full.tail(calc_days)   # 最適化に使う期間（直近1/2/3年）

最適化ループ（買い × 売りの総当たり）:
  buy_precalc × sell_precalc
    → β フィルタ（|net_beta| < target_beta）
    → 相関フィルタ（corr > target_corr）→ valid_plans
    → 相関NG → fallback_plans
    → 各プランに sharpe・calmar・score を計算
  → スコア降順ソート → 上位1件を session_state に保存

結果表示: st.session_state['results'] から読み出し → グラフ・注文レシピ表示
```

---

## HUFJPY の合成

Yahoo Finance に `HUFJPY=X` が存在しないため、`fetch_data()` 内で合成：

```
HUFJPY = USDJPY / USDHUF
```

「1ドル = X円」÷「1ドル = Yフォリント」=「1フォリント = X/Y 円」という計算。  
合成後は `USDHUF` 列を削除し、他の通貨ペアと同じように扱う。  
USDHUF のデータが取れない場合は HUFJPY 列が存在しないままになり、最適化ループ前の欠損チェックで検出・エラー表示される。

ブローカーによって 1lot の単位が異なる点に注意：HUFJPYは **1lot = 100,000通貨**（例：みんなのFX）。初期設定は `DEFAULT_LOT_UNIT` で管理。

---

## 認証

パスワードは `st.secrets["APP_PASSWORD"]` → 環境変数 `APP_PASSWORD` の順で取得（`_load_secret_password()` 関数）。どちらも未設定の場合は常にデモモード（通貨がMXN/TRYの2種のみに制限）。**ソースコードにパスワードを直接書かないこと。**

デプロイ先（Render）では Environment ページで `APP_PASSWORD` を環境変数として設定する。

---

## 開発上の注意

- **`df_full` は価格ではなく対数リターン**：変数名が誤解を招くが、`fetch_data()` が返す第1引数は `np.log(price).diff().dropna()` 済みのデータ。バックテストグラフでも `df_plot[ccy]` は対数リターンであり、`np.expm1()` で単純リターンに変換してから損益計算している。
- **`calc_calmar` のドキュメントと実装のズレ**：docstring に「/ capital」と記載があるが実装では省略。分子・分母が同じスケールのJPY値なので比率は同値（capitalが相殺される）。
- **`@st.cache_data` のキャッシュ**：データ確認のためにキャッシュを消したい場合はブラウザ右上の「Clear cache」から。
- **`session_state` と再計算**：計算ボタン押下後の結果は `st.session_state['results']` に格納される。サイドバー設定を変えても再度「計算スタート」を押すまで表示は更新されない。
- **TRY の長期リスク**：USD/TRYは長期的に右肩下がり（トルコリラ安継続）。シャープ・カルマー最適化によりシステムが自動的にTRY配分を抑制する設計だが、ユーザーが比率制限で上限を緩めた場合はこの限りではない。

---

## 運用の目安（実績ベース）

| 項目 | 参考値 |
|------|--------|
| 推奨実効レバレッジ | 16.6倍（証拠金維持率150%） |
| 目標β | ±0.05 以内 |
| 再計算タイミング | 2〜3か月に1回（TRY組み入れ時は毎月） |
| 調整トリガー | βが±0.05超、または評価損益が自己資金の±5%超 |
| 最大ドローダウン実績 | 約7%（2026年2月イランショック時：16.65%、2か月弱で回復） |
