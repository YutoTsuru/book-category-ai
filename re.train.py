import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from joblib import dump
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

csv_path = DATA_DIR / "clean_details.csv"
df = pd.read_csv(csv_path)

df = df.dropna(subset=["タイトル", "説明文"]).reset_index(drop=True)

df["text"] = (
    df["タイトル"].astype(str) + "。 " +
    df["説明文"].astype(str)
)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=200000
)

X = vectorizer.fit_transform(df["text"])

kmeans = KMeans(
    n_clusters=7,
    random_state=42,
    n_init=20
)

labels = kmeans.fit_predict(X)

dump(vectorizer, MODELS_DIR / "vectorizer.joblib")
dump(kmeans, MODELS_DIR / "kmeans.pkl")

df["cluster"] = labels
df[["タイトル", "cluster"]].to_csv(
    MODELS_DIR / "cluster_preview.csv",
    index=False,
    encoding="utf-8"
)

print("KMeans 学習完了")
