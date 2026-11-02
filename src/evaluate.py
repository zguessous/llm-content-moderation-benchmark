import os
import json
import glob
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from scipy.stats import pearsonr, spearmanr

from data_loader import TOXICITY_COLS

def compute_metrics(y_true, y_pred, threshold=0.5):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_corr, p_value = pearsonr(y_true, y_pred)
    spearman_corr = spearmanr(y_true, y_pred).statistic
    

    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    roc_auc = roc_auc_score(y_true_binary, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()

    return {
        "mse": float(mse),
        "mae": float(mae),
        "pearson_r": float(pearson_corr),
        "spearman_r": float(spearman_corr),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc_auc),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }

def main():
    summary_metrics = []
    gt_path = "results/data/sampled_n_200.csv"
    gt_df = pd.read_csv(gt_path)
    gt_dict = dict(zip(gt_df["id"], gt_df["target"]))
    gt_subtype_dict = {col: dict(zip(gt_df["id"], gt_df[col])) for col in TOXICITY_COLS}

    llm_outputs_paths = glob.glob("results/llm_outputs/*.json")

    for path in llm_outputs_paths:
        y_true = []
        y_pred = []
        y_true_subtypes = {col: [] for col in TOXICITY_COLS}
        y_pred_subtypes = {col: [] for col in TOXICITY_COLS}

        with open(path, "r") as f:
            records = json.load(f)

        if not records:
            continue

        provider = records[0]["provider"]
        model = records[0]["model"]
        technique = records[0]["technique"]
        include_subtypes = "subtypes" in records[0]["result"]
        row_entry = {
            "provider": provider,
            "model": model,
            "technique": technique,
            "include_subtypes": include_subtypes
        }
        
        for rec in records:
            rec_id = rec["id"]
            
            if rec_id in gt_dict:
                y_true.append(gt_dict[rec_id])
                if "result" not in rec or "toxicity_score" not in rec["result"]:
                    print(f"Warning: Missing toxicity_score in file {path}, record ID: {rec.get('id')}")
                y_pred.append(rec["result"]["toxicity_score"])

            if include_subtypes:
                if "result" not in rec or "subtypes" not in rec["result"]:
                    print(f"Warning: Missing toxicity_score in file {path}, record ID: {rec.get('id')}")
                pred_subtype_dict = rec["result"]["subtypes"]
                for col in TOXICITY_COLS:
                    if rec_id in gt_subtype_dict[col]:
                        y_true_subtypes[col].append(gt_subtype_dict[col][rec_id])
                        if col in pred_subtype_dict:
                            y_pred_subtypes[col].append(pred_subtype_dict[col])
                        else:
                            print(f"⚠️ Warning: Missing subtype key '{col}' in file: {path} | Record ID: {rec_id}")

        metrics = compute_metrics(y_true, y_pred)
        row_entry.update(metrics)
        if include_subtypes:
            for col in TOXICITY_COLS:
                subtypes_metrics = compute_metrics(y_true_subtypes[col], y_pred_subtypes[col])
                for k, v in subtypes_metrics.items():
                    row_entry[f"{col}_{k}"] = v

        summary_metrics.append(row_entry)

    summary_metrics_df = pd.DataFrame(summary_metrics)
    summary_metrics_df.to_csv("results/metrics/summary_metrics.csv", index=False)

if __name__ == "__main__":
    main()

