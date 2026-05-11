import re
import pickle
import hashlib
from pathlib import Path

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)

CSV_PATH  = "data/cleaned_medical_25mb.csv"
OUT_PATH  = "agents/triage_model.pkl"
TOP_N     = 20      # top medical categories to keep
TEST_SIZE = 0.20
STOP_WORDS  = set(stopwords.words("english"))
LEMMATIZER  = WordNetLemmatizer()
JUNK_RE     = re.compile(r"Â|â€™|â€œ|â€\x9d|â€\"|\\u00e2|\\u0080|\\u0099|\\xa0")
SPACE_RE    = re.compile(r"\s+")


def preprocess(text: str) -> str:
    text = JUNK_RE.sub(" ", str(text).lower())
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = SPACE_RE.sub(" ", text).strip()
    tokens = [
        LEMMATIZER.lemmatize(t)
        for t in text.split()
        if t not in STOP_WORDS and len(t) > 1
    ]
    return " ".join(tokens)

def main() -> None:
    print(f"Loading {CSV_PATH} …")
    df = pd.read_csv(CSV_PATH, encoding="latin1", on_bad_lines="skip")
    df = df.dropna(subset=["Patient", "Doctor", "Description"])

    # Mirror notebook: keep top-N medical categories
    top_cats = df["Description"].value_counts().head(TOP_N).index
    df = df[df["Description"].isin(top_cats)].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"  {len(df):,} rows across {TOP_N} categories")

    # Label encode
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Description"])

    # Preprocess patient questions
    print("Preprocessing text …")
    df["patient_clean"] = df["Patient"].apply(preprocess)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["patient_clean"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=42,
        stratify=df["label"],
    )

    # Pipeline: TF-IDF + LinearSVC (best performer from notebook)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=6000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("clf", LinearSVC(max_iter=2000, C=1.0)),
    ])

    print("Training LinearSVC pipeline …")
    pipeline.fit(X_train, y_train)

    preds  = pipeline.predict(X_test)
    report = classification_report(y_test, preds, target_names=le.classes_, zero_division=0)
    print("\nClassification Report:")
    print(report)

    # Bundle pipeline + label encoder for production use
    bundle = {
        "pipeline":      pipeline,
        "label_encoder": le,
        "categories":    list(le.classes_),
        "top_n":         TOP_N,
        "preprocess_fn": preprocess,
    }

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nSaved triage model → {OUT_PATH}")
    print(f"Categories: {list(le.classes_)}")


if __name__ == "__main__":
    main()
