import pandas as pd
import numpy as np

# -----------------------------
# Config
# -----------------------------
CSV_PATH = "/home/marco/Desktop/SAM3-exp/sam3/outputs/SAM3_HangCon_Prompt/hanging_object/score_0_5/hangcon_eval_exp3.csv"   # <-- cambia con il path reale
VIEWS = ["D1", "D2", "D3", "D4", "D8", "S"]

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Estrai la vista dal basename (prima parte prima dell'_')
df["view"] = df["basename"].str.split("_").str[0]

# Tieni solo le viste note (per sicurezza)
df = df[df["view"].isin(VIEWS)]

# -----------------------------
# Funzione metriche binarie
# -----------------------------
def compute_binary_metrics(y_true, y_pred):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / max(TP + TN + FP + FN, 1)
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# -----------------------------
# Calcolo metriche per vista
# -----------------------------
results = []

for view, g in df.groupby("view"):
    y_true = g["gt_has_object"]
    y_pred = g["pred_present"]

    metrics = compute_binary_metrics(y_true, y_pred)

    # mean IoU solo per:
    # - GT presente
    # - predizione presente
    # - iou_class2 non NaN
    valid_iou = g[
        (g["gt_has_object"] == 1) &
        (g["pred_present"] == 1) &
        (~g["iou_class2"].isna())
    ]["iou_class2"]

    mean_iou = valid_iou.mean() if len(valid_iou) > 0 else np.nan

    results.append({
        "view": view,
        "num_samples": len(g),
        "num_gt_positive": int(np.sum(g["gt_has_object"] == 1)),
        "num_pred_positive": int(np.sum(g["pred_present"] == 1)),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "mean_iou_hang_rope": mean_iou
    })

# -----------------------------
# Risultati finali
# -----------------------------
df_view_metrics = pd.DataFrame(results).sort_values("view")

print(df_view_metrics.to_string(index=False, float_format="%.4f"))

# (opzionale) salva su CSV
df_view_metrics.to_csv("/home/marco/Desktop/SAM3-exp/sam3/outputs/SAM3_HangCon_Prompt/hangcon_eval_per_category.csv", index=False)
