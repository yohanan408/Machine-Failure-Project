# Machine Failure Production Pipeline - Maintained by John
"""Machine Failure Prediction Pipeline

This script reproduces the analysis and modelling performed in the original
`Machine Failure.ipynb` notebook, but it is organized as a reusable, production‑
ready Python module.

Key steps
-----------
1. **Load data** – read the CSV file.
2. **Preprocess** – rename columns, drop the unused identifier, encode the
   categorical *Type* column and split the data into features/target.
3. **Resample** – under‑sample the majority class to obtain a balanced training
   set.
4. **Train models** – Logistic Regression, Decision Tree (with tuned hyper‑
   parameters), Random Forest and k‑Nearest Neighbours.
5. **Evaluate** – classification report, confusion matrix and ROC‑AUC for each
   model.

All heavy‑weight visualisation code from the notebook has been omitted to keep
the script lightweight for production use.  If visualisation is required, the
functions can be extended or called from a separate notebook.
"""

from __future__ import annotations

import pathlib
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_data(csv_path: str | pathlib.Path) -> pd.DataFrame:
    """Load the machine‑failure CSV file.

    Parameters
    ----------
    csv_path: str or Path
        Path to ``machine failure.csv``.

    Returns
    -------
    pd.DataFrame
        Raw dataframe as stored in the CSV file.
    """
    return pd.read_csv(csv_path)


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Clean and encode the raw dataframe.

    The steps replicate the notebook:
    * rename columns to snake_case
    * drop the ``UDI`` column (identifier not needed for modelling)
    * encode the categorical ``Type`` column with ``LabelEncoder``

    Returns
    -------
    x: DataFrame
        Feature matrix (all columns except ``Product_ID``, ``Type`` and the
        target ``Machine_failure``).
    y: Series
        Target vector – ``Machine_failure``.
    """
    # rename columns
    df = df.rename(
        columns={
            "Product ID": "Product_ID",
            "Air temperature [K]": "Air_temperature",
            "Process temperature [K]": "Process_temperature",
            "Rotational speed [rpm]": "Rotational_speed",
            "Torque [Nm]": "Torque",
            "Tool wear [min]": "Tool_wear",
            "Machine failure": "Machine_failure",
        }
    )
    # drop identifier column
    if "UDI" in df.columns:
        df = df.drop(columns="UDI")

    # encode categorical Type column
    encoder = LabelEncoder()
    df["Type"] = encoder.fit_transform(df["Type"])

    # split into features / target
    y = df["Machine_failure"]
    x = df.drop(columns=["Product_ID", "Type", "Machine_failure"])
    return x, y


def split_and_resample(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.02,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train‑test split followed by random under‑sampling of the
    training set to balance classes.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    rus = RandomUnderSampler(random_state=random_state)
    x_train_rus, y_train_rus = rus.fit_resample(x_train, y_train)
    return x_train_rus, x_test, y_train_rus, y_test


def correlation_feature_selection(data: pd.DataFrame, threshold: float = 0.7) -> set:
    """Return a set of column names whose absolute correlation with any other
    column exceeds *threshold*.

    The original notebook used this to drop ``Process_temperature``.
    """
    corr_matrix = data.corr()
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                to_drop.add(corr_matrix.columns[i])
    return to_drop


def train_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, object]:
    """Train five classifiers and return them in a dictionary.

    The models mirror the notebook implementations:
    * ``logistic`` – Logistic Regression
    * ``decision_tree`` – DecisionTreeClassifier with tuned hyper‑parameters
    * ``random_forest`` – RandomForestClassifier (default settings)
    * ``knn`` – KNeighborsClassifier (k=3)
    * ``gradient_boost`` – GradientBoostingClassifier
    """
    models: Dict[str, object] = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(x_train, y_train)
    models["logistic"] = lr

    # Decision Tree – use the best params from the notebook
    dt = DecisionTreeClassifier(
        max_depth=None,
        max_features=7,
        min_samples_leaf=8,
        criterion="gini",
        random_state=42,
    )
    dt.fit(x_train, y_train)
    models["decision_tree"] = dt

    # Random Forest – default hyper‑parameters
    rf = RandomForestClassifier(random_state=42)
    rf.fit(x_train, y_train)
    models["random_forest"] = rf

    # k‑Nearest Neighbours (k=3)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    models["knn"] = knn

    # XGBoost Classifier – default parameters, can be tuned later
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(x_train, y_train)
    models["xgboost"] = xgb

    # Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    gbc.fit(x_train, y_train)
    models["gradient_boost"] = gbc

    return models


def evaluate(
    models: Dict[str, object],
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Print classification reports, confusion matrices and ROC‑AUC values.

    The function writes the results to ``stdout`` – suitable for logging or
    interactive use.  It deliberately avoids heavy visualisation to keep the
    script lightweight; however, the ROC curves can be plotted by extending the
    function with Matplotlib if required.
    """
    results = []
    for name, model in models.items():
        preds = model.predict(x_test)
        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds)
        fpr, tpr, _ = roc_curve(y_test, preds)
        model_auc = auc(fpr, tpr)
        results.append((name, report, cm, model_auc))

    # Print a tidy summary
    for name, report, cm, model_auc in results:
        print(f"{'='*20}\nModel: {name}\n{'='*20}")
        print("Classification report:")
        # Convert the dict back to a string for readability
        print(pd.DataFrame(report).transpose())
        print("\nConfusion matrix:")
        print(cm)
        print(f"\nROC‑AUC: {model_auc:.4f}\n")


# ---------------------------------------------------------------------------
# Main execution path
# ---------------------------------------------------------------------------

def main(csv_path: str | pathlib.Path = "/mnt/c/Users/John/Documents/machine failure.csv") -> None:
    # 1. Load raw data
    raw_df = load_data(csv_path)

    # 2. Preprocess / encode
    x, y = preprocess(raw_df)

    # 3. Train‑test split + under‑sampling
    x_train_rus, x_test, y_train_rus, y_test = split_and_resample(x, y)

    # 4. Optional correlation‑based feature removal (as in the notebook)
    drop_cols = correlation_feature_selection(x_train_rus, threshold=0.7)
    if drop_cols:
        print(f"Dropping correlated columns: {drop_cols}")
        x_train_rus = x_train_rus.drop(columns=list(drop_cols))
        x_test = x_test.drop(columns=list(drop_cols))

    # 5. Train models
    models = train_models(x_train_rus, y_train_rus)

    # 6. Evaluate
    evaluate(models, x_test, y_test)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Layout / comment block – placeholders for future pipeline stages
# ---------------------------------------------------------------------------
# End of file
