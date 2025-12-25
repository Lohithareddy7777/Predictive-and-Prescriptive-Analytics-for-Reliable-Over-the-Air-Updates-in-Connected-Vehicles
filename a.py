# ============================================
# OTA Update Failure Prediction & Optimization
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# -----------------------------
# 1. Synthetic Data Generation
# -----------------------------

np.random.seed(42)
N = 5000

data = {
    "battery_soc": np.random.randint(20, 100, N),                 # %
    "signal_strength_rssi": np.random.randint(-120, -60, N),      # dBm
    "vehicle_is_moving": np.random.choice([0, 1], N, p=[0.8, 0.2]),
    "prior_fail_count": np.random.poisson(0.5, N),
    "update_size_mb": np.random.randint(100, 3000, N),
    "time_of_day": np.random.choice(
        ["Night", "Morning", "Afternoon", "Evening"],
        N,
        p=[0.35, 0.25, 0.20, 0.20]
    )
}

df = pd.DataFrame(data)

# ---- Failure Probability Logic (REALISTIC ASSUMPTIONS) ----
failure_prob = (
    0.35 * (df["battery_soc"] < 40).astype(int) +
    0.30 * (df["signal_strength_rssi"] < -95).astype(int) +
    0.20 * (df["vehicle_is_moving"] == 1).astype(int) +
    0.25 * (df["update_size_mb"] > 1500).astype(int) +
    0.15 * (df["prior_fail_count"] > 1).astype(int) +
    0.10 * (df["time_of_day"].isin(["Morning", "Evening"])).astype(int)
)

failure_prob = np.clip(failure_prob, 0, 0.9)
df["update_failure"] = np.random.binomial(1, failure_prob)

df.to_csv("synthetic_ota_data.csv", index=False)

print("\nDataset generated.")
print(df["update_failure"].value_counts(normalize=True))

# -----------------------------
# 2. Model Training
# -----------------------------

feature_cols = [
    "battery_soc",
    "signal_strength_rssi",
    "vehicle_is_moving",
    "prior_fail_count",
    "update_size_mb",
    "time_of_day"
]

target_col = "update_failure"

X = df[feature_cols].copy()
y = df[target_col]

# One-hot encoding for categorical feature
X = pd.get_dummies(X, columns=["time_of_day"], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=100,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

print("\n--- Training Model ---")
model.fit(X_train, y_train)
print("--- Training Complete ---")

# -----------------------------
# 3. Model Evaluation
# -----------------------------

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n### Model Evaluation ###")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# 4. Risk Probability Analysis
# -----------------------------

analysis_df = X_test[["battery_soc", "signal_strength_rssi"]].copy()
analysis_df["prob_failure"] = y_pred_proba
analysis_df["true_failure"] = y_test.values

analysis_df["soc_bin"] = pd.cut(
    analysis_df["battery_soc"],
    bins=[0, 40, 70, 100],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

analysis_df["rssi_bin"] = pd.cut(
    analysis_df["signal_strength_rssi"],
    bins=[-130, -95, -85, -60],
    labels=["Weak", "Moderate", "Strong"],
    include_lowest=True
)

risk_table = (
    analysis_df
    .groupby(["soc_bin", "rssi_bin"])
    .agg(
        avg_failure_probability=("prob_failure", "mean"),
        sample_count=("prob_failure", "count")
    )
    .round(3)
)

# -----------------------------
# 5. Decision Rules (Model-Derived)
# -----------------------------

def recommendation(prob):
    if prob > 0.6:
        return "Defer OTA"
    elif prob > 0.35:
        return "Delay / Retry Later"
    else:
        return "Allow OTA"

risk_table["recommendation"] = risk_table["avg_failure_probability"].apply(recommendation)

print("\n### OTA Scheduling Decision Table ###")
print(risk_table)

# -----------------------------
# 6. Business Impact Estimation
# -----------------------------

baseline_failure_rate = y_test.mean()

safe_updates = analysis_df[
    (analysis_df["battery_soc"] > 70) &
    (analysis_df["signal_strength_rssi"] > -90)
]

optimized_failure_rate = safe_updates["prob_failure"].mean()

print("\n### Business Impact ###")
print(f"Baseline Failure Rate: {baseline_failure_rate:.2f}")
print(f"Optimized Failure Rate: {optimized_failure_rate:.2f}")
print(
    f"Failure Reduction: "
    f"{(baseline_failure_rate - optimized_failure_rate) / baseline_failure_rate:.2%}"
)

# -----------------------------
# 7. Feature Importance Plot
# -----------------------------

importance = model.get_booster().get_score(importance_type="weight")
importance_df = pd.Series(importance).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importance_df.plot(kind="barh")
plt.title("Feature Importance for OTA Failure Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
