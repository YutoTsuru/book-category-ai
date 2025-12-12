import streamlit as st
import requests

API_URL = "https://book-category-api.onrender.com/predict"

cluster_label_map = {
    0: "漫画・ライトノベル",
    1: "IT技術",
    2: "文芸・小説・シリーズ",
    3: "エッセイ・評論",
    4: "一般教養・解説書",
    5: "大学受験（赤本・過去問）",
    6: "高校/中学受験（年度版ガイド）"
}

st.set_page_config(page_title="書籍ジャンル分類AI", layout="centered")

st.markdown(
    """
    <h1 style='text-align:center; font-size:42px;'>
        書籍ジャンル分類AI
    </h1>
    """,
    unsafe_allow_html=True
)

title = st.text_input("タイトル")
col1, col2 = st.columns([1, 1])

with col1:
    author = st.text_input("著者")
with col2:
    publisher = st.text_input("出版社")

desc = st.text_area("説明文", height=200)

if st.button("分類する"):
    text = f"{title}。{author}。{publisher}。{desc[:200]}"

    response = requests.post(API_URL, json={"text": text})

    if response.status_code != 200:
        st.error(f"API Error: {response.status_code}")
    else:
        data = response.json()
        st.write(data)
        pred = data.get("cluster")

        if not isinstance(pred, int):
            pred = None

        genre = cluster_label_map.get(pred, "不明")

        st.markdown(
            f"""
            <div style="
                background:#1f2937;
                color:white;
                padding:24px;
                border-radius:12px;
                text-align:center;
                font-size:24px;
                font-weight:bold;
            ">
            🎯 推定ジャンル：{genre}
            </div>
            """,
            unsafe_allow_html=True
        )
