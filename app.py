import streamlit as st
import requests

API_URL = "https://web-production-f66ba.up.railway.app/predict"

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
col1, col2 = st.columns(2)

with col1:
    author = st.text_input("著者")
with col2:
    publisher = st.text_input("出版社")

desc = st.text_area("説明文", height=200)

if st.button("分類する"):
    text = " ".join([title, author, publisher, desc]).strip()

    if not text:
        st.warning("何か入力してね！")
        st.stop()

    try:
        response = requests.post(API_URL, json={"text": text}, timeout=10)
        response.raise_for_status()
        result = response.json()
        genre = result["label"]

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
                推定ジャンル：<b>{genre}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error("APIとの通信に失敗しました")
        st.code(str(e))
