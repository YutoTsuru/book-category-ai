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
    text = f"{title}。{author}。{publisher}。{desc}"
    response = requests.post(API_URL, json={"text": text})
    genre = response.json()["genre"]

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