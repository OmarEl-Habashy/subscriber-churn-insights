import sys, json, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score, classification_report, confusion_matrix)
import xgboost as xgb

TARGET_COL = "churn"
ID_COL = "user_id"

def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def split_data(df: pd.DataFrame, test_size=0.2, val_size=0.15, random_state=42):
    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
    x = df[feature_cols]
    y = df[TARGET_COL]

    x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, test_size = test_size, stratify=y, random_state=random_state)

    val_ratio = val_size / (1-test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=val_ratio, stratify=y_train_full, random_state=random_state)
    return x_train, x_val, x_test, y_train, y_val, y_test, feature_cols

def evaluate(name, y_true, y_pred, y_prob):
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    print(f"\n--- {name} ---")
    print(f"F1 : {f1:.4f}\n")
    print(f"ROC-AUC : {roc_auc:.4f}\n")
    print(f"PR-AUC : {pr_auc:.4f}\n")
    print(classification_report(y_true, y_pred, digits=4))
    print("\n")
    print("Confusion Matrix : " + "\n", confusion_matrix(y_true, y_pred))
    return {"f1" : f1, "roc_auc" : roc_auc, "pr_auc" : pr_auc}

def train_logistic_regression(x_train, y_train, x_val, scaler):
    x_train_s = scaler.transform(x_train)
    x_val_s = scaler.transform(x_val)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(x_train_s, y_train)
    val_prob = model.predict_proba(x_val_s)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)
    return model, val_pred, val_prob

def train_svm(x_train, y_train, x_val, scaler):
    x_train_s = scaler.transform(x_train)
    x_val_s = scaler.transform(x_val)
    model = SVC(kernel="linear", class_weight="balanced", probability=True, random_state=42)
    model.fit(x_train_s, y_train)
    val_prob = model.predict_proba(x_val_s)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)
    return model, val_pred, val_prob

def train_xgboost(x_train, y_train, x_val, y_val):
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators = 300,
        max_depth = 4,
        learning_rate = 0.05,
        subsample = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = scale_pos_weight,
        eval_metric = "logloss",
        early_stopping_rounds = 20,
        random_state = 42,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    val_prob = model.predict_proba(x_val)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)
    return model, val_pred, val_prob


def cross_validate_models(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {
        "logistic_regression": {"f1": [], "roc_auc": [], "pr_auc": []},
        "linear_svm": {"f1": [], "roc_auc": [], "pr_auc": []},
        "xgboost": {"f1": [], "roc_auc": [], "pr_auc": []},
    }
 
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
 
        scaler = StandardScaler()
        scaler.fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)
 
        print(f"\nFold {fold_i} (test n={len(test_idx)}, churners={int(y_te.sum())})")
 
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
        lr.fit(X_tr_s, y_tr)
        lr_prob = lr.predict_proba(X_te_s)[:, 1]
        lr_pred = (lr_prob >= 0.5).astype(int)
        _record(results["logistic_regression"], y_te, lr_pred, lr_prob, "  LR    ")
 
        svm = SVC(kernel="linear", class_weight="balanced", probability=True, random_state=random_state)
        svm.fit(X_tr_s, y_tr)
        svm_prob = svm.predict_proba(X_te_s)[:, 1]
        svm_pred = (svm_prob >= 0.5).astype(int)
        _record(results["linear_svm"], y_te, svm_pred, svm_prob, "  SVM   ")
 
        scale_pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=random_state,
        )
        xgb_model.fit(X_tr, y_tr)
        xgb_prob = xgb_model.predict_proba(X_te)[:, 1]
        xgb_pred = (xgb_prob >= 0.5).astype(int)
        _record(results["xgboost"], y_te, xgb_pred, xgb_prob, "  XGB   ")
 
    print("\n" + "=" * 60)
    print("5-FOLD CV COMPARISON — mean +/- std across all folds")
    print("=" * 60)
    summary = {}
    for model_name, metrics in results.items():
        summary[model_name] = {
            metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            for metric, vals in metrics.items()
        }
        s = summary[model_name]
        print(f"{model_name:22s}  F1: {s['f1']['mean']:.3f}+/-{s['f1']['std']:.3f}   "
              f"ROC-AUC: {s['roc_auc']['mean']:.3f}+/-{s['roc_auc']['std']:.3f}   "
              f"PR-AUC: {s['pr_auc']['mean']:.3f}+/-{s['pr_auc']['std']:.3f}")
 
    winner = max(summary, key=lambda k: summary[k]["pr_auc"]["mean"])
    print(f"\n>>> Winner by mean CV PR-AUC: {winner}")
    return summary, winner
 
 
def _record(bucket, y_true, y_pred, y_prob, label):
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    bucket["f1"].append(f1)
    bucket["roc_auc"].append(roc_auc)
    bucket["pr_auc"].append(pr_auc)
    print(f"{label} F1={f1:.3f}  ROC-AUC={roc_auc:.3f}  PR-AUC={pr_auc:.3f}")
 



def main(path: str):
    df = load_features(path)
    print(f"Loaded {len(df)} rows, churn rate {df[TARGET_COL].mean():.1%}")
 
    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
 
    print("\n=== 5-fold cross-validation comparison (LR vs SVM vs XGBoost) ===")
    cv_summary, cv_winner = cross_validate_models(df[feature_cols], df[TARGET_COL])
 

    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_data(df)
    print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
 
    scaler = StandardScaler()
    scaler.fit(X_train)
 
    if cv_winner == "logistic_regression":
        model, _, _ = train_logistic_regression(X_train, y_train, X_val, scaler)
        needs_scaling = True
    elif cv_winner == "linear_svm":
        model, _, _ = train_svm(X_train, y_train, X_val, scaler)
        needs_scaling = True
    else:
        model, _, _ = train_xgboost(X_train, y_train, X_val, y_val)
        needs_scaling = False
 
    X_test_input = scaler.transform(X_test) if needs_scaling else X_test
    test_prob = model.predict_proba(X_test_input)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    test_metrics = evaluate(f"{cv_winner} (single-split test, for reference only)", y_test, test_pred, test_prob)
 
    # --- persist artifacts ---
    joblib.dump(model, "artifacts/model.pkl")
    joblib.dump(scaler, "artifacts/scaler.pkl")
    with open("artifacts/feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open("artifacts/model_info.json", "w") as f:
        json.dump({
            "model_name": cv_winner,
            "needs_scaling": needs_scaling,
            "cv_summary": cv_summary,
            "single_split_test_metrics_reference_only": test_metrics,
        }, f, indent=2)
 
    print("\nSaved artifacts/model.pkl, scaler.pkl, feature_columns.json, model_info.json")
    winner_cv = cv_summary[cv_winner]
    print(f"\n>>> Model selected by 5-fold CV: {cv_winner}")
    print(f">>> REPORT THIS — F1: {winner_cv['f1']['mean']:.3f}+/-{winner_cv['f1']['std']:.3f}  "
          f"ROC-AUC: {winner_cv['roc_auc']['mean']:.3f}+/-{winner_cv['roc_auc']['std']:.3f}  "
          f"PR-AUC: {winner_cv['pr_auc']['mean']:.3f}+/-{winner_cv['pr_auc']['std']:.3f}")
 
 
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sparkify_features.csv"
    main(path)