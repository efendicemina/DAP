import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

MODEL_PATH = "models/intent_classifier.joblib"
DATASET_PATH = "mpr_dataset_final/intent_test.csv"

model = joblib.load(MODEL_PATH)

test_df = pd.read_csv(DATASET_PATH)

X_test = test_df["text"]
y_test = test_df["label"]

preds = model.predict(X_test)

print("\n======================")
print("INTENT MODEL ANALYSIS")
print("======================")

print(f"\nSamples: {len(test_df)}")

print(f"\nAccuracy: {accuracy_score(y_test, preds):.4f}")
print(f"Macro F1: {f1_score(y_test, preds, average='macro'):.4f}")
print(f"Weighted F1: {f1_score(y_test, preds, average='weighted'):.4f}")

print("\nClassification report:\n")
print(classification_report(y_test, preds))

print("\nConfusion matrix:\n")

labels = sorted(y_test.unique())

cm = confusion_matrix(y_test, preds, labels=labels)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)

# confidence analysis
probs = model.predict_proba(X_test)

max_probs = probs.max(axis=1)

print("\nConfidence analysis:")
print(f"Average confidence: {max_probs.mean():.4f}")
print(f"Min confidence: {max_probs.min():.4f}")
print(f"Max confidence: {max_probs.max():.4f}")

low_conf = test_df.copy()
low_conf["confidence"] = max_probs
low_conf["predicted"] = preds

low_conf = low_conf.sort_values("confidence").head(20)

print("\nLowest confidence predictions:\n")

for _, row in low_conf.iterrows():
    print("=" * 80)
    print("TEXT:", row["text"])
    print("TRUE:", row["label"])
    print("PRED:", row["predicted"])
    print("CONF:", round(row["confidence"], 4))