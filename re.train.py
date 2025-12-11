import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from joblib import dump
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

csv_path = DATA_DIR / "clean_details.csv"
df = pd.read_csv(csv_path)

df = df.dropna(subset=["タイトル", "説明文", "カテゴリ"])
df = shuffle(df, random_state=42).reset_index(drop=True)

df["text"] = (
    df["タイトル"].astype(str) + "。 " +
    df["著者"].astype(str) + "。 " +
    df["出版社"].astype(str) + "。 " +
    df["説明文"].astype(str)
)

X = df["text"].values
y_text = df["カテゴリ"].values

le = LabelEncoder()
y = le.fit_transform(y_text)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_features=200000)),
    ("clf", LinearSVC(C=1.0))
])

train_sizes, train_scores, val_scores = learning_curve(
    pipe, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.2,1.0,5),
    scoring="f1_macro", n_jobs=-1, shuffle=True, random_state=42
)

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
report = classification_report(y_test, y_pred, target_names=le.classes_, digits=4)
cm = confusion_matrix(y_test, y_pred)

with open(REPORTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=220)
plt.close()

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)
plt.figure()
plt.plot(train_sizes, train_mean, marker="o", label="train")
plt.plot(train_sizes, val_mean, marker="s", label="cv")
plt.xlabel("train size")
plt.ylabel("macro F1")
plt.title("learning curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "learning_curve.png", dpi=220)
plt.close()

dump(pipe, MODELS_DIR / "genre_pipeline.joblib")
dump(le, MODELS_DIR / "label_encoder.joblib")

tfidf = pipe.named_steps["tfidf"]
clf = pipe.named_steps["clf"]
feature_names = np.array(tfidf.get_feature_names_out())

if clf.coef_.ndim == 2:
    for i, cls in enumerate(le.classes_):
        top_idx = np.argsort(clf.coef_[i])[-15:][::-1]
        terms = feature_names[top_idx]
        weights = clf.coef_[i][top_idx]
        plt.figure(figsize=(6,4))
        plt.barh(range(len(terms)), weights[::-1])
        plt.yticks(range(len(terms)), terms[::-1])
        plt.xlabel("weight")
        plt.title(f"top terms: {cls}")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"top_terms_{i}_{cls}.png", dpi=220)
        plt.close()

X_all = tfidf.transform(X)
np.savez_compressed(
    MODELS_DIR / "tfidf_corpus.npz",
    data=X_all.data, indices=X_all.indices,
    indptr=X_all.indptr, shape=X_all.shape
)

meta = {
    "id": df["isbn13"].astype(str).tolist(),
    "title": df["タイトル"].astype(str).tolist(),
    "genre": df["カテゴリ"].astype(str).tolist()
}
dump(meta, MODELS_DIR / "corpus_meta.joblib")
