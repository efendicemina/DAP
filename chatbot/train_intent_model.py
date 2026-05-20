import joblib
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

DATA_DIR = Path("mpr_dataset_final")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

train_df = pd.read_csv(DATA_DIR / "intent_train.csv")
val_df = pd.read_csv(DATA_DIR / "intent_val.csv")
test_df = pd.read_csv(DATA_DIR / "intent_test.csv")

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

model.fit(train_df["text"], train_df["label"])

for name, df in [("VALIDATION", val_df), ("TEST", test_df)]:
    preds = model.predict(df["text"])

    print("\n====================")
    print(name)
    print("====================")
    print("Accuracy:", accuracy_score(df["label"], preds))
    print(classification_report(df["label"], preds))

joblib.dump(model, MODEL_DIR / "intent_classifier.joblib")

print("\nModel saved to models/intent_classifier.joblib")