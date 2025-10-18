# app.py
# ------------------------------------------------------------
# 日本語の書籍タイトル＋説明文からジャンルを分類する簡易アプリ
# - 学習ノートで保存したモデルに両対応:
#   * genre_pipeline.joblib + labels.npy（推奨）
#   * genre_pipeline.joblib + label_encoder.joblib（旧）
# - 類似本検索は tfidf_corpus.npz / corpus_meta.joblib があれば自動でON
# ------------------------------------------------------------

from pathlib import Path
import numpy as np
import streamlit as st
from joblib import load
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# パス設定
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

PIPE_PATH = MODELS_DIR / "genre_pipeline.joblib"
LE_JOBLIB_PATH = MODELS_DIR / "label_encoder.joblib"   # 旧保存形式
LABELS_NPY_PATH = MODELS_DIR / "labels.npy"            # 現在の保存形式

CORPUS_NPZ_PATH = MODELS_DIR / "tfidf_corpus.npz"      # 類似本（任意）
CORPUS_META_PATH = MODELS_DIR / "corpus_meta.joblib"   # 類似本（任意）

# ------------------------------------------------------------
# 便利関数
# ------------------------------------------------------------
def load_label_encoder() -> LabelEncoder:
    """labels.npy か label_encoder.joblib のどちらかを読み、LabelEncoder を返す。"""
    if LE_JOBLIB_PATH.exists():
        le = load(LE_JOBLIB_PATH)
        if not isinstance(le, LabelEncoder):
            raise TypeError("label_encoder.joblib は LabelEncoder ではありません。")
        return le

    if LABELS_NPY_PATH.exists():
        classes = np.load(LABELS_NPY_PATH, allow_pickle=True)
        le = LabelEncoder().fit(classes)
        return le

    raise FileNotFoundError("ラベル情報が見つかりません（labels.npy または label_encoder.joblib が必要）")


def get_step(pipeline, *names):
    """
    Pipeline.named_steps から、候補名のいずれかを返す。
    学習時の書き方の違い（('tfidf', ...) vs make_pipeline(TfidfVectorizer)）に対応。
    """
    for n in names:
        if n in pipeline.named_steps:
            return pipeline.named_steps[n]
    return None


def softmax_like(margin: np.ndarray) -> np.ndarray:
    """SVM の decision_function を確率風にスケーリング（便宜的）。"""
    x = margin - np.max(margin)
    ex = np.exp(x)
    p = ex / np.sum(ex)
    return p


def concat_text(title: str, description: str) -> str:
    title = (title or "").strip()
    desc = (description or "").strip()
    return (title + " " + desc).strip()


# ------------------------------------------------------------
# モデルのロード
# ------------------------------------------------------------
st.set_page_config(page_title="Book Genre Classifier", layout="wide")

st.sidebar.title("モデル状態")
try:
    pipe = load(PIPE_PATH)
    st.sidebar.success(f"✅ Pipeline: {PIPE_PATH.name}")
except Exception as e:
    st.sidebar.error("❌ Pipeline が読めませんでした。学習ノートで保存してください。")
    st.exception(e)
    st.stop()

try:
    le: LabelEncoder = load_label_encoder()
    st.sidebar.success("✅ Labels: labels.npy / label_encoder.joblib")
except Exception as e:
    st.sidebar.error("❌ ラベル情報がありません。")
    st.exception(e)
    st.stop()

# 類似本（任意）
similarity_ready = False
tfidf_corpus = None
corpus_meta = None
try:
    if CORPUS_NPZ_PATH.exists() and CORPUS_META_PATH.exists():
        tfidf_corpus = np.load(CORPUS_NPZ_PATH)
        corpus_meta = load(CORPUS_META_PATH)  # {"titles": [...], "genres": [...], ...} を想定
        similarity_ready = "X" in tfidf_corpus and "titles" in corpus_meta
        if similarity_ready:
            st.sidebar.info("🔎 類似本コーパス: 有効")
        else:
            st.sidebar.warning("🔎 類似本コーパスのフォーマットが不完全です。")
    else:
        st.sidebar.write("🔎 類似本コーパス: なし（任意機能）")
except Exception as e:
    st.sidebar.warning("🔎 類似本を読み込めませんでした（任意）。")
    st.sidebar.code(str(e))

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("📚 日本語・書籍ジャンル分類（SVM）")

with st.expander("ℹ️ 使い方", expanded=False):
    st.write(
        "- 上に学習済みモデル（`models/genre_pipeline.joblib`）が必要です。\n"
        "- ラベルは `models/labels.npy`（推奨）か `models/label_encoder.joblib` のどちらかが必要です。\n"
        "- 類似本は任意で `tfidf_corpus.npz` と `corpus_meta.joblib` があれば表示されます。"
    )

col1, col2 = st.columns(2)
title = col1.text_input("タイトル", value="", placeholder="例）機械学習入門")
description = col2.text_area("説明（著者/出版社/概要など）", height=130, placeholder="例）著者◯◯／出版社△△／…")

run = st.button("🔮 予測する", type="primary")

# ------------------------------------------------------------
# 予測
# ------------------------------------------------------------
if run:
    user_text = concat_text(title, description)
    if not user_text:
        st.warning("タイトルか説明を入力してください。")
        st.stop()

    # そのまま pipeline で推論
    try:
        y_pred = pipe.predict([user_text])[0]
    except Exception as e:
        st.error("推論でエラーが発生しました。")
        st.exception(e)
        st.stop()

    # decision_function があれば、確信度も出す
    try:
        margins = pipe.decision_function([user_text])  # shape: (1, n_classes) or list-like (ovo)
        if isinstance(margins, list):  # 2値や OVO の場合の簡易対処
            margins = margins[0]
        probs = softmax_like(np.array(margins).ravel())
        top_idx = np.argsort(probs)[::-1]
        class_names = le.classes_
        st.subheader("🎯 予測結果")
        st.markdown(f"**予測ジャンル:** `{y_pred}`")

        # 上位5件の確信度を表示
        k = min(5, len(class_names))
        top_table = [
            {"rank": i + 1, "genre": class_names[idx], "score": float(probs[idx])}
            for i, idx in enumerate(top_idx[:k])
        ]
        st.table(top_table)

    except Exception:
        # decision_function が無いモデルでも最低限表示
        st.subheader("🎯 予測結果")
        st.markdown(f"**予測ジャンル:** `{y_pred}`")
        st.caption("（このモデルは確信度スコアを計算できません）")

    # --------------------------------------------------------
    # 類似本（任意）
    # --------------------------------------------------------
    if similarity_ready:
        st.markdown("---")
        st.subheader("🔎 類似している本（コーパス内）")

        # ベクトル化器を取得（両対応）
        vect = get_step(pipe, "tfidf", "tfidfvectorizer")
        if vect is None:
            st.info("ベクトル化器が見つからないため、類似本検索はスキップしました。")
        else:
            try:
                # 入力をベクトル化
                q_vec = vect.transform([user_text])

                # コーパス行列
                X_corpus = tfidf_corpus["X"]  # csr_matrix を想定
                sims = cosine_similarity(q_vec, X_corpus).ravel()

                # 同ジャンル優先でフィルタ（同点なら全体から）
                titles = corpus_meta.get("titles", [])
                genres = corpus_meta.get("genres", [])
                same_genre_idx = [i for i, g in enumerate(genres) if g == y_pred] if genres else []

                def topk_from(indices, k=10):
                    if not indices:
                        return []
                    idx = np.argsort(sims[indices])[::-1][:k]
                    return [indices[i] for i in idx]

                picks = topk_from(same_genre_idx, k=10)
                if len(picks) < 5:
                    # 足りなければ全体からも補完
                    extra = np.argsort(sims)[::-1][:10]
                    extra = [i for i in extra if i not in picks]
                    picks = (picks + extra)[:10]

                rows = []
                for i in picks:
                    rows.append({
                        "title": titles[i] if i < len(titles) else f"id:{i}",
                        "genre": genres[i] if i < len(genres) else "-",
                        "similarity": float(sims[i]),
                    })

                if rows:
                    st.dataframe(rows, use_container_width=True)
                else:
                    st.write("類似本を表示できませんでした。")

            except Exception as e:
                st.info("類似本検索でエラーが発生しました（任意機能）。")
                st.exception(e)

# ------------------------------------------------------------
# フッター
# ------------------------------------------------------------
st.markdown("---")
st.caption("Model: LinearSVC + TF-IDF（学習ノートからエクスポート） / UI: Streamlit")
