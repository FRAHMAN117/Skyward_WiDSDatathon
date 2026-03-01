
"""
WiDS Global Datathon 2026 - Tree Model Baseline
------------------------------------------------
This script trains simple tree-based classifiers for each time horizon
(12h, 24h, 48h, 72h) and produces a valid Kaggle submission.

Usage:
    python baseline_tree_survival.py

Expected files in the same folder:
    - train.csv
    - test.csv
    - sample_submission.csv

Output:
    - submission.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

HORIZONS = [12, 24, 48, 72]

def main():
    print("Loading data...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sub = pd.read_csv("sample_submission.csv")

    id_col = "event_id"

    # Target columns
    time_col = "time_to_hit_hours"
    event_col = "event"

    # Feature columns
    drop_cols = [id_col, time_col, event_col]
    features = [c for c in train.columns if c not in drop_cols]

    X_train = train[features].copy()
    X_test = test[features].copy()

    # Simple preprocessing
    print("Filling missing values...")
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    preds = []

    for H in HORIZONS:
        print(f"Training tree model for horizon {H}h...")

        # Binary label: hit by horizon H
        y = ((train[event_col] == 1) & (train[time_col] <= H)).astype(int)

        model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

        model.fit(X_train, y)

        prob = model.predict_proba(X_test)[:, 1]
        preds.append(prob)

    preds = np.vstack(preds).T

    # Enforce monotonicity
    preds = np.maximum.accumulate(preds, axis=1)

    sub["prob_12h"] = preds[:, 0]
    sub["prob_24h"] = preds[:, 1]
    sub["prob_48h"] = preds[:, 2]
    sub["prob_72h"] = preds[:, 3]

    print("Saving submission.csv")
    sub.to_csv("submission.csv", index=False)

    print("Done! Upload submission.csv to Kaggle.")

if __name__ == "__main__":
    main()
