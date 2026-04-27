import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

coil_df = pd.read_csv("shape_data_coil.txt", sep=r'\s+')
globule_df = pd.read_csv("shape_data_globule.txt", sep=r'\s+')

coil_df["label"] = 1
globule_df["label"] = 0

df = pd.concat([coil_df, globule_df], ignore_index=True)
print(f"Total samples: {len(df)}  |  Coil: {len(coil_df)}  |  Globule: {len(globule_df)}")
print(df.head())

X = df[["Rg", "Re"]].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "SVM (RBF kernel)":    SVC(kernel="rbf", probability=True, random_state=42)
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

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Globule", "Coil"],
                yticklabels=["Globule", "Coil"])
    ax.set_title(f"{name}\nAccuracy: {res['accuracy']*100:.1f}%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Decision Boundaries (Rg vs Re)", fontsize=14, fontweight="bold")

X_all_scaled = scaler.transform(X)
x_min, x_max = X_all_scaled[:, 0].min() - 0.5, X_all_scaled[:, 0].max() + 0.5
y_min, y_max = X_all_scaled[:, 1].min() - 0.5, X_all_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

for ax, (name, res) in zip(axes, results.items()):
    model = res["model"]
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    contour = ax.contourf(xx, yy, Z, levels=50, cmap="RdYlBu", alpha=0.8)
    plt.colorbar(contour, ax=ax, label="P(Coil)")

    scatter = ax.scatter(X_all_scaled[:, 0], X_all_scaled[:, 1],
                         c=y, cmap="bwr", edgecolors="k",
                         linewidths=0.5, s=20, zorder=5)

    ax.contour(xx, yy, Z, levels=[0.5], colors="black",
               linewidths=2, linestyles="--")

    ax.set_title(f"{name}")
    ax.set_xlabel("Rg (scaled)")
    ax.set_ylabel("Re (scaled)")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
               markersize=8, label='Coil (ε=0.5)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=8, label='Globule (ε=2.0)'),
        Line2D([0], [0], color='black', linewidth=2,
               linestyle='--', label='θ-point boundary')
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig("decision_boundary.png", dpi=150)
plt.show()

print("\n" + "="*50)
print("θ-POINT IDENTIFICATION")
print("Scanning Rg at mean Re — finding where P(Coil) crosses 0.5")
print("="*50)

mean_Re    = df["Re"].mean()
rg_scan    = np.linspace(df["Rg"].min(), df["Rg"].max(), 500)
re_fixed   = np.full_like(rg_scan, mean_Re)
scan_input = scaler.transform(np.column_stack([rg_scan, re_fixed]))

best_model_name = max(results, key=lambda k: results[k]["accuracy"])
best_model      = results[best_model_name]["model"]
proba_coil      = best_model.predict_proba(scan_input)[:, 1]

crossing_idx = np.argmin(np.abs(proba_coil - 0.5))
theta_Rg     = rg_scan[crossing_idx]

print(f"Using best model: {best_model_name}")
print(f"Mean Re used for scan: {mean_Re:.3f}")
print(f"Estimated θ-point Rg: {theta_Rg:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(rg_scan, proba_coil, color="steelblue", linewidth=2, label="P(Coil)")
plt.axhline(0.5, color="red",   linestyle="--", linewidth=1.5, label="50% threshold")
plt.axvline(theta_Rg, color="green", linestyle="--",
            linewidth=1.5, label=f"θ-point Rg ≈ {theta_Rg:.3f}")
plt.fill_between(rg_scan, proba_coil, 0.5,
                 where=(proba_coil > 0.5), alpha=0.15, color="blue",  label="Coil region")
plt.fill_between(rg_scan, proba_coil, 0.5,
                 where=(proba_coil < 0.5), alpha=0.15, color="red",   label="Globule region")
plt.xlabel("Rg", fontsize=12)
plt.ylabel("P(Coil)", fontsize=12)
plt.title("θ-Point Identification via Classification Probability", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("theta_point.png", dpi=150)
plt.show()

print("\nAll plots saved as PNG files in the current directory.")


