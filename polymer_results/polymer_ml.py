import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── CONFIG ───────────────────────────────────────────────────
DATA_DIR         = "epsilon_results"   # folder with all shape_data files
THETA_EPSILON    = 1.5                 # boundary: below = Coil, above = Globule
                                       # adjust this after seeing your results
OUTPUT_DIR       = "ml_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. LOAD ALL EPSILON FILES ─────────────────────────────────
all_files = sorted(glob.glob(f"{DATA_DIR}/shape_data_eps_*.txt"))

if not all_files:
    raise FileNotFoundError(f"No files found in {DATA_DIR}/. Check your folder path.")

print(f"Found {len(all_files)} epsilon files\n")

frames = []
for filepath in all_files:
    # Extract epsilon value from filename
    # e.g. shape_data_eps_0.5.txt → 0.5
    fname   = os.path.basename(filepath)
    epsilon = float(fname.replace("shape_data_eps_", "").replace(".txt", ""))

    df = pd.read_csv(filepath, sep=r'\s+')
    df["epsilon"] = epsilon
    df["label"]   = 1 if epsilon < THETA_EPSILON else 0  # Coil=1, Globule=0
    frames.append(df)

# Combine all into one dataframe
full_df = pd.concat(frames, ignore_index=True)

print(f"Total samples : {len(full_df)}")
print(f"Coil samples  : {(full_df['label']==1).sum()}  (epsilon < {THETA_EPSILON})")
print(f"Globule samples: {(full_df['label']==0).sum()}  (epsilon >= {THETA_EPSILON})")
print(f"\nEpsilon values used:\n{sorted(full_df['epsilon'].unique())}\n")

# Save combined dataset
full_df.to_csv(f"{OUTPUT_DIR}/combined_dataset.csv", index=False)
print(f"Combined dataset saved → {OUTPUT_DIR}/combined_dataset.csv")


# ── 2. PLOT Rg vs EPSILON (shows collapse transition) ────────
epsilon_summary = full_df.groupby("epsilon")["Rg"].mean().reset_index()
epsilon_summary.to_csv(f"{OUTPUT_DIR}/rg_vs_epsilon.csv", index=False)

plt.figure(figsize=(9, 5))
plt.plot(epsilon_summary["epsilon"], epsilon_summary["Rg"],
         marker="o", color="steelblue", linewidth=2, markersize=5)
plt.axvline(THETA_EPSILON, color="red", linestyle="--",
            linewidth=2, label=f"θ-boundary (ε={THETA_EPSILON})")
plt.xlabel("Epsilon (ε)", fontsize=12)
plt.ylabel("Mean Rg", fontsize=12)
plt.title("Mean Radius of Gyration vs Epsilon\n(Coil-to-Globule Transition)", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rg_vs_epsilon.png", dpi=150)
plt.show()
print("Plot saved → rg_vs_epsilon.png")

# ── 3. FEATURES & LABELS ─────────────────────────────────────
X = full_df[["Rg", "Re"]].values
y = full_df["label"].values

# ── 4. TRAIN/TEST SPLIT ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. FEATURE SCALING ───────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 6. TRAIN MODELS ──────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "SVM (RBF kernel)":    SVC(kernel="rbf", probability=True, random_state=42, C=10)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    results[name] = {"model": model, "accuracy": acc, "y_pred": y_pred}
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Globule", "Coil"]))

# ── 7. CONFUSION MATRICES ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Confusion Matrices — Epsilon Sweep Dataset", fontsize=14, fontweight="bold")

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Globule","Coil"],
                yticklabels=["Globule","Coil"])
    ax.set_title(f"{name}\nAccuracy: {res['accuracy']*100:.1f}%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150)
plt.show()

# ── 8. DECISION BOUNDARY ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Decision Boundaries — Epsilon Sweep", fontsize=14, fontweight="bold")

X_all_scaled = scaler.transform(X)
x_min = X_all_scaled[:, 0].min() - 0.5
x_max = X_all_scaled[:, 0].max() + 0.5
y_min = X_all_scaled[:, 1].min() - 0.5
y_max = X_all_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

for ax, (name, res) in zip(axes, results.items()):
    model = res["model"]
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

    contour = ax.contourf(xx, yy, Z, levels=50, cmap="RdYlBu", alpha=0.8)
    plt.colorbar(contour, ax=ax, label="P(Coil)")
    ax.scatter(X_all_scaled[:, 0], X_all_scaled[:, 1],
               c=y, cmap="bwr", edgecolors="k", linewidths=0.4, s=10, zorder=5)
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2, linestyles="--")
    ax.set_title(name)
    ax.set_xlabel("Rg (scaled)")
    ax.set_ylabel("Re (scaled)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/decision_boundary.png", dpi=150)
plt.show()

# ── 9. θ-POINT SCAN ──────────────────────────────────────────
best_model_name = max(results, key=lambda k: results[k]["accuracy"])
best_model      = results[best_model_name]["model"]

mean_Re   = full_df["Re"].mean()
rg_scan   = np.linspace(full_df["Rg"].min(), full_df["Rg"].max(), 1000)
re_fixed  = np.full_like(rg_scan, mean_Re)
scan_input = scaler.transform(np.column_stack([rg_scan, re_fixed]))
proba_coil = best_model.predict_proba(scan_input)[:, 1]

crossing_idx = np.argmin(np.abs(proba_coil - 0.5))
theta_Rg     = rg_scan[crossing_idx]

print(f"\n{'='*50}")
print(f"θ-POINT RESULT")
print(f"Best model  : {best_model_name}")
print(f"θ-point Rg  : {theta_Rg:.4f}")
print(f"(Using mean Re = {mean_Re:.3f} for scan)")

# Save theta point result
with open(f"{OUTPUT_DIR}/theta_point_result.txt", "w") as f:
    f.write(f"Best model: {best_model_name}\n")
    f.write(f"Theta-point Rg: {theta_Rg:.4f}\n")
    f.write(f"Mean Re used: {mean_Re:.4f}\n")

plt.figure(figsize=(9, 5))
plt.plot(rg_scan, proba_coil, color="steelblue", linewidth=2, label="P(Coil)")
plt.axhline(0.5, color="red",   linestyle="--", linewidth=1.5, label="50% threshold")
plt.axvline(theta_Rg, color="green", linestyle="--",
            linewidth=1.5, label=f"θ-point Rg ≈ {theta_Rg:.3f}")
plt.fill_between(rg_scan, proba_coil, 0.5,
                 where=(proba_coil > 0.5), alpha=0.15, color="blue", label="Coil region")
plt.fill_between(rg_scan, proba_coil, 0.5,
                 where=(proba_coil < 0.5), alpha=0.15, color="red",  label="Globule region")
plt.xlabel("Rg", fontsize=12)
plt.ylabel("P(Coil)", fontsize=12)
plt.title(f"θ-Point Identification — {best_model_name}", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/theta_point.png", dpi=150)
plt.show()

# ── 10. FIND θ-POINT EPSILON ─────────────────────────────────
# For each epsilon, take the last 20% of steps (equilibrated)
# and get mean Rg and Re → feed into model → get P(Coil)
# Find where P(Coil) crosses 0.5 → that epsilon IS the θ-point

print("\n" + "="*50)
print("FINDING θ-POINT EPSILON")
print("="*50)

epsilon_proba = []

for filepath in all_files:
    fname   = os.path.basename(filepath)
    epsilon = float(fname.replace("shape_data_eps_", "").replace(".txt", ""))

    df_eps = pd.read_csv(filepath, sep=r'\s+')

    # Use only last 20% of steps = equilibrated/settled state
    last_20_percent = int(len(df_eps) * 0.8)
    df_eq = df_eps.iloc[last_20_percent:]

    mean_Rg = df_eq["Rg"].mean()
    mean_Re = df_eq["Re"].mean()

    # Scale and predict
    sample        = scaler.transform([[mean_Rg, mean_Re]])
    prob_coil     = best_model.predict_proba(sample)[0][1]

    epsilon_proba.append({
        "epsilon":   epsilon,
        "mean_Rg":   mean_Rg,
        "mean_Re":   mean_Re,
        "P_coil":    prob_coil,
        "P_globule": 1 - prob_coil,
        "state":     "Coil" if prob_coil >= 0.5 else "Globule"
    })

    print(f"ε = {epsilon:.1f}  |  Rg = {mean_Rg:.3f}  |  P(Coil) = {prob_coil:.3f}  →  {('Coil' if prob_coil >= 0.5 else 'Globule')}")

# Convert to dataframe
ep_df = pd.DataFrame(epsilon_proba).sort_values("epsilon")
ep_df.to_csv(f"{OUTPUT_DIR}/epsilon_theta_scan.csv", index=False)

# Find the exact crossing point
# Look for where state flips from Coil to Globule
theta_epsilon = None
for i in range(len(ep_df) - 1):
    row_now  = ep_df.iloc[i]
    row_next = ep_df.iloc[i + 1]
    if row_now["state"] == "Coil" and row_next["state"] == "Globule":
        # Interpolate between the two epsilon values
        e1, p1 = row_now["epsilon"],  row_now["P_coil"]
        e2, p2 = row_next["epsilon"], row_next["P_coil"]
        theta_epsilon = e1 + (0.5 - p1) * (e2 - e1) / (p2 - p1)
        break

if theta_epsilon:
    print(f"\n{'='*50}")
    print(f"★ θ-POINT EPSILON = {theta_epsilon:.3f}")
    print(f"  Below ε = {theta_epsilon:.3f}  → Coil state")
    print(f"  Above ε = {theta_epsilon:.3f}  → Globule state")
    print(f"{'='*50}")
else:
    print("\nCould not find a clean crossing — try adjusting THETA_EPSILON boundary")

# Plot P(Coil) vs Epsilon
plt.figure(figsize=(9, 5))
plt.plot(ep_df["epsilon"], ep_df["P_coil"],
         marker="o", color="steelblue", linewidth=2, markersize=6, label="P(Coil)")
plt.axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="50% threshold")
if theta_epsilon:
    plt.axvline(theta_epsilon, color="green", linestyle="--",
                linewidth=2, label=f"θ-point ε ≈ {theta_epsilon:.3f}")
plt.fill_between(ep_df["epsilon"], ep_df["P_coil"], 0.5,
                 where=(ep_df["P_coil"] >= 0.5), alpha=0.15,
                 color="blue", label="Coil region")
plt.fill_between(ep_df["epsilon"], ep_df["P_coil"], 0.5,
                 where=(ep_df["P_coil"] < 0.5), alpha=0.15,
                 color="red", label="Globule region")
plt.xlabel("Epsilon (ε)", fontsize=12)
plt.ylabel("P(Coil)", fontsize=12)
plt.title("θ-Point: P(Coil) vs Epsilon", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/theta_point_epsilon.png", dpi=150)
plt.show()

# Save final result
with open(f"{OUTPUT_DIR}/theta_point_result.txt", "w") as f:
    f.write(f"Best model: {best_model_name}\n")
    f.write(f"Theta-point Epsilon: {theta_epsilon:.3f}\n")
    f.write(f"Interpretation: Below e={theta_epsilon:.3f} = Coil, Above = Globule\n")

print(f"\nAll outputs saved in: {OUTPUT_DIR}/")
print("Done!")

print(f"\nAll outputs saved in: {OUTPUT_DIR}/")
print("Done!")


