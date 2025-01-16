import pandas as pd
from scipy.stats import ks_2samp
from sklearn import metrics


def get_ks(df: pd.DataFrame, proba_col: str, true_value_col: str):
    
    # Recover each class
    class0 = df[df[true_value_col] == 0]
    class1 = df[df[true_value_col] == 1]
    ks = ks_2samp(class0[proba_col], class1[proba_col])
    return ks.statistic

def get_report_metrics(df: pd.DataFrame, proba_col: str, true_value_col: str, base: str):
    ks = get_ks(df, proba_col, true_value_col)
    acc = metrics.accuracy_score(df[true_value_col], df[proba_col].round())
    auc = metrics.roc_auc_score(df[true_value_col], df[proba_col])
    precison = metrics.precision_score(df[true_value_col], df[proba_col].round())
    recall = metrics.recall_score(df[true_value_col], df[proba_col].round())
    report = {
        f"{base} ks": ks,
        f"{base} accuracy": acc,
        f"{base} auc": auc,
        f"{base} precision": precison,
        f"{base} recall": recall
    }
    
    return report
    