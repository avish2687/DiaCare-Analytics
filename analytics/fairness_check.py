import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score


def group_fairness_metrics(df, y_pred, sensitive_column):
    """
    Computes basic group fairness metrics:
    - count
    - accuracy
    - recall
    - positive prediction rate

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe (unprocessed)
    y_pred : np.array
        Model predictions (0/1)
    sensitive_column : str
        Column name (e.g. 'gender', 'race', 'age')

    Returns
    -------
    dict
        Fairness metrics per group
    """

    results = {}

    if sensitive_column not in df.columns:
        raise ValueError(f"Column '{sensitive_column}' not found in dataframe")

    groups = df[sensitive_column].fillna("Unknown")

    for group in groups.unique():
        idx = groups == group

        if idx.sum() == 0:
            continue

        y_true = None
        if "readmitted" in df.columns:
            y_true = df.loc[idx, "readmitted"].apply(
                lambda x: 1 if x == "<30" else 0
            )

        preds = y_pred[idx]

        metrics = {
            "count": int(idx.sum()),
            "positive_rate": float(np.mean(preds))
        }

        if y_true is not None:
            metrics["accuracy"] = float(accuracy_score(y_true, preds))
            metrics["recall"] = float(recall_score(y_true, preds))
        else:
            metrics["accuracy"] = None
            metrics["recall"] = None

        results[group] = metrics

    return results
