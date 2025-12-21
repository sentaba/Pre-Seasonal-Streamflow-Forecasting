"""
PCR_analysis.py

Principal Component Regression (PCR) for pre-seasonal AMJ streamflow forecasting
in the Michipicoten River Watershed, Canada.

Author: Sent
Date: 2025

Requirements:
- pandas, numpy, matplotlib, scikit-learn, scipy

Outputs:
- PCA components, loadings, scree/cumulative variance plots
- Predictor influence ranking
- Observed vs predicted plots
- Contingency table and comprehensive results CSV
- Summary.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# ==========================================================
# 1. SET PATHS (relative to repository)
# ==========================================================
output_dir = "figures/PCR_outputs"
os.makedirs(output_dir, exist_ok=True)

predictor_file = "data/processed/PCR_JFM_Predictors.csv"
predictand_file = "data/processed/streamflow_02BD002_AMJ.csv"

# ==========================================================
# 2. LOAD DATA
# ==========================================================
predictors = pd.read_csv(predictor_file)
predictand = pd.read_csv(predictand_file)

predictors.rename(columns={predictors.columns[0]: "Year"}, inplace=True)
predictand.rename(columns={predictand.columns[0]: "Year"}, inplace=True)

# ==========================================================
# 3. ALIGN YEARS
# ==========================================================
merged = pd.merge(predictors, predictand, on="Year", how="inner")
merged.to_csv(os.path.join(output_dir, "Aligned_Predictors_and_Predictand.csv"), index=False)

X = merged.drop(columns=["Year", predictand.columns[1]])
y = merged[predictand.columns[1]]

# ==========================================================
# 3a. PLOT PREDICTORS AND PREDICTAND TOGETHER
# ==========================================================
plt.figure(figsize=(12,6))
X_norm = (X - X.mean()) / X.std()
y_norm = (y - y.mean()) / y.std()
for col in X_norm.columns:
    plt.plot(merged["Year"], X_norm[col], label=col, linestyle='--', alpha=0.7)
plt.plot(merged["Year"], y_norm, label="Streamflow (AMJ)", color="black", linewidth=2)
plt.xlabel("Year")
plt.ylabel("Normalized Value")
plt.title("Predictors and Predictand (Normalized) Over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Predictors_and_Predictand_TimeSeries.png"), dpi=300)
plt.close()

# ==========================================================
# 4. STANDARDIZE PREDICTORS
# ==========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================================
# 5. PCA
# ==========================================================
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Save PCA components
pd.DataFrame(
    X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
).to_csv(os.path.join(output_dir, "PCA_Components.csv"), index=False)

# Individual and cumulative variance (%)
ind_var = pca.explained_variance_ratio_ * 100
cum_var = np.cumsum(ind_var)

# Select number of PCs for >=80% cumulative variance
threshold = 80.0
selected_pc = np.argmax(cum_var >= threshold) + 1
selected_pc = min(selected_pc, 3)  # fix to 3 if desired

# Scree plot
plt.figure(figsize=(7,5))
plt.plot(range(1, len(ind_var)+1), ind_var, marker='o')
plt.axvline(selected_pc, color='red', linestyle='--', label=f'Selected PC ({selected_pc})')
plt.xlabel("Principal Component")
plt.ylabel("Percentage of Variance")
plt.title("Scree Plot (Individual Variance in %)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Scree_Plot.png"), dpi=300)
plt.close()

# Cumulative variance plot
plt.figure(figsize=(7,5))
plt.plot(range(1, len(cum_var)+1), cum_var, marker='s', color='green')
plt.axhline(threshold, color='red', linestyle='--', label='80% Threshold')
plt.axvline(selected_pc, color='purple', linestyle='--', label=f'Selected PC ({selected_pc})')
plt.xlabel("Principal Component")
plt.ylabel("Cumulative Variance (%)")
plt.title("PCA Cumulative Variance Explained")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "PCA_Cumulative_Variance.png"), dpi=300)
plt.close()

# PCA loadings
loadings = pd.DataFrame(
    pca.components_.T,
    index=X.columns,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)]
)
loadings.to_csv(os.path.join(output_dir, "PCA_Loadings.csv"))

# ==========================================================
# 6. SELECT PCs AND FIT PCR
# ==========================================================
n_components = selected_pc
X_pca_sel = X_pca[:, :n_components]

model = LinearRegression()
model.fit(X_pca_sel, y)
y_pred = model.predict(X_pca_sel)

# Coefficients
coef_df = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(n_components)],
    "Coefficient": model.coef_
})
coef_df.to_csv(os.path.join(output_dir, "PCR_Model_Details.csv"), index=False)

# Predictor influence ranking
true_coef = loadings.iloc[:, :n_components].values @ model.coef_
influence_df = pd.DataFrame({
    "Predictor": X.columns,
    "Influence": true_coef
}).sort_values(by="Influence", ascending=False)
influence_df.to_csv(os.path.join(output_dir, "Predictor_influence_ranking.csv"), index=False)

plt.figure(figsize=(9,6))
plt.bar(influence_df["Predictor"], influence_df["Influence"])
plt.xticks(rotation=90)
plt.title("Predictor Influence Ranking (PCR)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Predictor_Influence_Ranking.png"), dpi=300)
plt.close()

# ==========================================================
# 7. MODEL PERFORMANCE
# ==========================================================
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
R_value, p_value = pearsonr(y, y_pred)
nse = 1 - (np.sum((y_pred - y)**2) / np.sum((y - np.mean(y))**2))

perf_df = pd.DataFrame({
    "RMSE":[rmse],
    "MAE":[mae],
    "Pearson_R":[R_value],
    "p_value":[p_value],
    "NSE":[nse]
})
perf_df.to_csv(os.path.join(output_dir, "PCR_Model_Performance.csv"), index=False)

# ==========================================================
# 8. SCATTER AND TIME SERIES PLOTS
# ==========================================================
plt.figure(figsize=(7,5))
plt.scatter(y, y_pred, edgecolor="black", alpha=0.7)
m, b = np.polyfit(y, y_pred, 1)
plt.plot(y, m*y + b, color='red')
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.title("Observed vs Predicted")
plt.text(0.95, 0.05, f"R={R_value:.3f}", transform=plt.gca().transAxes,
         horizontalalignment='right', verticalalignment='bottom', fontsize=12)
plt.savefig(os.path.join(output_dir, "Scatter_Observed_vs_Predicted.png"), dpi=300)
plt.close()

plt.figure(figsize=(10,5))
plt.plot(merged["Year"], y, marker="o", label="Observed")
plt.plot(merged["Year"], y_pred, marker="s", label="Predicted")
plt.title("Observed vs Predicted Streamflow (Time Series)")
plt.xlabel("Year")
plt.ylabel("Streamflow")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_dir, "Observed_vs_Predicted_Streamflow.png"), dpi=300)
plt.close()

# ==========================================================
# 9. CONTINGENCY TABLE & COMPREHENSIVE RESULTS
# ==========================================================
t1 = np.percentile(y, 33.33)
t2 = np.percentile(y, 66.67)
def classify(v):
    if v < t1: return "Below"
    elif v < t2: return "Normal"
    else: return "Above"

obs_cat = y.apply(classify)
pred_cat = pd.Series(y_pred).apply(classify)
cont_table = pd.crosstab(obs_cat, pred_cat)
cont_table.to_csv(os.path.join(output_dir, "Contingency_Table.csv"))

results = pd.DataFrame({
    "Year": merged["Year"],
    "Observed": y,
    "Predicted": y_pred,
    "Observed_Category": obs_cat,
    "Predicted_Category": pred_cat
})
results.to_csv(os.path.join(output_dir, "PCR_Comprehensive_Results_AMJ.csv"), index=False)

# ==========================================================
# 10. SUMMARY FILE
# ==========================================================
with open(os.path.join(output_dir, "Summary.txt"), "w") as f:
    f.write("=== PCR FINAL SUMMARY ===\n")
    f.write("Project Title: A Principal Component Regression Model for Pre-Seasonal Streamflow Forecasting in the Michipicoten River Watershed, Canada\n\n")
    f.write("Selected PCs (>=80% cumulative variance): {}\n\n".format(n_components))
    f.write(perf_df.to_string())
    f.write("\n\nContingency Table:\n")
    f.write(cont_table.to_string())
    f.write("\n\nRegression Equation (PCR Model):\n")
    eq = f"Q_AMJ = {model.intercept_:.3f} "
    for i, coef in enumerate(model.coef_):
        eq += f"+ ({coef:.3f} * PC{i+1}) "
    f.write(eq.strip() + "\n")

print("\nAll outputs successfully generated in:", output_dir)
