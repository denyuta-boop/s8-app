# S8戦略 自動最適化ツール

FX キャリートレード戦略（S8戦略）のポートフォリオを自動最適化する Streamlit アプリです。

## 機能

- **通貨ペア自動選択**: 買い候補（MXN/ZAR/PLN/TRY/CZK/HUF）× 売り候補（USD/CHF/EUR）の全組み合わせをスキャン
- **3つの最適化モード**:
  - シャープレシオ最大（推奨）: リスク調整後リターンを最大化
  - スワップ最大（旧方式）: 日次スワップ収益を重視
  - カスタム加重: スワップ・シャープ・カルマーを任意の比率で合成
- **リスク制御**: β（対USD）と相関係数によるフィルタリング、通貨ごとの比率制限
- **バックテスト**: 過去1〜3年の損益推移・ドローダウンをグラフ表示
- **注文レシピ**: 採用プランの推奨ロット数を自動計算

## ローカル実行

```bash
# 依存パッケージをインストール
pip install -r requirements.txt

# パスワードを環境変数にセット（未設定の場合はデモモードのみ）
export APP_PASSWORD="your_password"

# 起動
streamlit run app.py
```

## 環境変数

| 変数名 | 説明 |
|--------|------|
| `APP_PASSWORD` | フルアクセスパスワード。未設定の場合は常にデモモード（通貨選択が制限される） |

## デプロイ（Render）

1. Render ダッシュボードの **Environment** ページで `APP_PASSWORD` を設定
2. `main` ブランチへの push で自動デプロイ

## 技術スタック

- [Streamlit](https://streamlit.io/) — UI フレームワーク
- [yfinance](https://github.com/ranaroussi/yfinance) — 為替データ取得（Yahoo Finance）
- [Plotly](https://plotly.com/python/) — インタラクティブグラフ
- [SciPy](https://scipy.org/) — ベータ値計算（線形回帰）
