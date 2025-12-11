import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# ====== モデル読み込み ======
with open("models/kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("models/label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

model = SentenceTransformer("intfloat/multilingual-e5-small")

# ====== UI セットアップ ======
st.set_page_config(page_title="書籍ジャンル分類AI", layout="centered")

st.markdown(
    """
    <h1 style='text-align:center; font-size:42px;'>
        📚 書籍ジャンル分類AI
    </h1>
    """,
    unsafe_allow_html=True
)

# ---- 入力フォーム ----
title = st.text_input("タイトル")
col1, col2 = st.columns(2)

with col1:
    author = st.text_input("著者")
with col2:
    publisher = st.text_input("出版社")

desc = st.text_area("説明文", height=200)

# ====== 分類処理 ======
if st.button("分類する"):

    # 入力チェック
    text = " ".join([title, author, publisher, desc]).strip()

    if not text:
        st.warning("何か入力してね！")
        st.stop()

    # sentence-transformers で埋め込み生成
    emb = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)

    # KMeans クラスタ番号
    cluster_id = int(kmeans.predict(emb)[0])

    # クラスタ → ラベル名
    genre = label_map.get(cluster_id, "不明")

    # ====== 結果表示 ======
    st.markdown(
        f"""
        <div style="
            background:#1f2937;
            color:white;
            padding:20px;
            border-radius:12px;
            text-align:center;
            font-size:28px;
            margin-top:30px;
        ">
            🎯 推定ジャンル：<b>{genre}</b>
        </div>
        """,
        unsafe_allow_html=True
    )
