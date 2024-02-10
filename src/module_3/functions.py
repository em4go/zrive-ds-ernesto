import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import LogitResults

import pickle


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int or None = None,
) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=test_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def test_variables(X: pd.DataFrame, y: pd.Series) -> LogitResults:
    X_sm = sm.add_constant(X)
    model_statmodels = sm.Logit(y, X_sm)
    result_statmodels = model_statmodels.fit()
    return result_statmodels


def get_best_treshold(y_test: np.array, y_pred_prob: np.array) -> float:
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

    f1_scores = 2 * (precision * recall) / (precision + recall)

    best_f1_index = np.argmax(f1_scores)

    best_threshold = thresholds[best_f1_index]

    return best_threshold


def print_metrics(y_test: np.array, y_test_pred: np.array) -> None:
    print("Accuracy:", accuracy_score(y_test, y_test_pred))

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

    print("Classification Report:\n", classification_report(y_test, y_test_pred))


def plot_roc_curve(y_test: np.array, y_pred_prob: np.array) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:0.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])  # 1.02 to make the upper limit visible
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curve(y_test: np.array, y_pred_prob: np.array) -> None:
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    average_precision = average_precision_score(y_test, y_pred_prob)

    plt.figure()
    plt.plot(
        recall,
        precision,
        color="blue",
        lw=2,
        label=f"Precision-Recall curve (auc = {average_precision:0.2f})",
    )
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall curve")
    plt.legend(loc="upper right")
    plt.show()


def filter_orders_by_n_products(df: pd.DataFrame, n_products: int) -> pd.DataFrame:
    products_ordered = df[df["outcome"] == 1]
    orders_len = products_ordered.groupby("order_id")["variant_id"].count()
    orders_over_n_products = orders_len[orders_len >= n_products].index
    filtered = df[df["order_id"].isin(orders_over_n_products)]
    return filtered


def add_useful_variables(df: pd.DataFrame) -> pd.DataFrame:
    products_sorted = df.sort_values(
        by=["user_id", "order_date"], ascending=[True, False]
    )

    last_ordered_products = (
        products_sorted.groupby(["user_id", "variant_id"]).first().reset_index()
    )

    times_ordered = (
        df.groupby(["user_id", "variant_id"])["outcome"]
        .sum()
        .reset_index()
        .rename(columns={"outcome": "times_ordered"})
    )

    last_ordered_products = last_ordered_products.merge(
        times_ordered, on=["user_id", "variant_id"], how="left"
    )

    last_ordered_products["days_to_purchase_variant_id_norm"] = (
        last_ordered_products["avg_days_to_buy_variant_id"]
        - last_ordered_products["days_since_purchase_variant_id"]
    ) / last_ordered_products["std_days_to_buy_variant_id"]

    last_ordered_products["days_to_purchase_variant_id_if_ordered_before"] = (
        last_ordered_products["days_since_purchase_variant_id"]
        * last_ordered_products["ordered_before"]
    )

    last_ordered_products["days_to_purchase_product_type_norm"] = (
        last_ordered_products["avg_days_to_buy_product_type"]
        - last_ordered_products["days_since_purchase_product_type"]
    ) / last_ordered_products["std_days_to_buy_product_type"]

    return last_ordered_products


def transform_dataset_to_fit_model(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    df = filter_orders_by_n_products(df, 5)
    df = add_useful_variables(df)

    return df


def get_user_variant_data(
    user_id: int,
    variant_id: int,
    df: pd.DataFrame,
) -> pd.DataFrame:
    X = df.query(f"user_id == {user_id} and variant_id == {variant_id}")
    return X


def get_possible_models() -> list[list[str]]:
    from possible_models import (
        MODEL_1,
        MODEL_2,
        MODEL_3,
        MODEL_4,
        MODEL_5,
    )

    all_models = [MODEL_1, MODEL_2, MODEL_3, MODEL_4, MODEL_5]
    return all_models


def get_model(model_name) -> LogisticRegression:
    all_models = get_possible_models()

    with open(f"./{model_name}_info.txt", "r") as info_file:
        info = info_file.read()
        info = info.split(",")
        model_index = int(info[0])
        model_threshold = float(info[1])
    model_variables = all_models[model_index]

    with open(f"{model_name}.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    return model, model_variables, model_threshold


def get_prediction(
    model: LogisticRegression,
    X: pd.DataFrame,
    threshold: float,
) -> np.array:
    y_pred_prob = model.predict_proba(X)[:, 1]
    y_pred = [1 if prob > threshold else 0 for prob in y_pred_prob]
    return y_pred[0]
