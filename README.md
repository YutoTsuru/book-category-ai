# Book Category AI

書籍のタイトルや説明文からジャンルを自動分類するAIアプリケーションです。
Python + scikit-learn + Streamlit を使用しています。

## ディレクトリ構成
- `data/` : 書籍データ (CSV, 100件以上)
- `models/` : 学習済みモデル、ラベルエンコーダ
- `reports/` : グラフや評価レポート
- `notebooks/` : データ分析用ノートブック
- `train.py` : 学習・評価・グラフ出力
- `app.py` : デモアプリ (Streamlit)

## 実行手順
```bash
python -m venv .venv
source .venv/bin/activate   # Windowsは .venv\Scripts\activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```
