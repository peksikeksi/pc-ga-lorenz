"""
Lorenz System - Comparative Analysis of Numerical Integration Methods
----------------------------------------------------------------------
This script compares four Lorenz system integration approaches:
1. High-accuracy RK45 baseline
2. Predictor-Corrector (ABM2, Euler bootstrap)
3. GA-optimized step size (ABM2 + GA-h)
4. GA-optimized coefficients (ABM2 + GA-params)

All results are quantitatively evaluated using RMSE and MAE metrics.
Visualizations include time-series overlays, error evolution, and summary bars.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# --- Load results ---
base = np.load("results/lorenz_baseline.npz")
pc = np.load("results/lorenz_pc_abm2_euler.npz")
ga_h = np.load("results/lorenz_pc_ga_optimized.npz")
ga_m = np.load("results/lorenz_pc_ga_multi.npz")

# Extract baseline trajectory
t_ref = base["t"]
x_ref, y_ref, z_ref = base["x"], base["y"], base["z"]
ref_traj = np.vstack((x_ref, y_ref, z_ref)).T

# Load methods
methods = {"ABM2 (Euler)": pc, "GA-optimized step": ga_h, "GA-optimized coeffs": ga_m}


# --- Helper metrics ---
def compute_metrics(t_ref, ref, t, y):
    """Interpolates to reference timeline and computes RMSE, MAE."""
    interp = np.array([np.interp(t_ref, t, y[:, i]) for i in range(3)]).T
    diff = interp - ref
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    return rmse, mae, diff


metrics = {}
for label, data in methods.items():
    t = data["t"]
    y = np.vstack((data["x"], data["y"], data["z"])).T
    rmse, mae, diff = compute_metrics(t_ref, ref_traj, t, y)
    metrics[label] = {"rmse": rmse, "mae": mae, "diff": diff}

# --- Print summary table ---
print("\n=== Quantitative Comparison ===")
print(f"{'Method':30} | {'RMSE':>10} | {'MAE':>10}")
print("-" * 55)
for label, vals in metrics.items():
    print(f"{label:30} | {vals['rmse']:10.4e} | {vals['mae']:10.4e}")

# --- Relative Improvement vs ABM2 ---
base_rmse = metrics["ABM2 (Euler)"]["rmse"]
base_mae = metrics["ABM2 (Euler)"]["mae"]

print("\n=== Relative Improvement over ABM2 (Euler) ===")
print(f"{'Method':30} | {'RMSE Δ%':>10} | {'MAE Δ%':>10}")
print("-" * 55)
for label, vals in metrics.items():
    if label == "ABM2 (Euler)":  # skip baseline
        continue
    rmse_improv = 100 * (base_rmse - vals["rmse"]) / base_rmse
    mae_improv = 100 * (base_mae - vals["mae"]) / base_mae
    metrics[label]["rmse_improv"] = rmse_improv
    metrics[label]["mae_improv"] = mae_improv
    print(f"{label:30} | {rmse_improv:10.2f}% | {mae_improv:10.2f}%")

# --- Visualization Section ---

# Time-series comparison for x(t)
plt.figure(figsize=(9, 5))
plt.plot(t_ref, x_ref, label="RK45 baseline", color="black", lw=1.2)
colors = ["orange", "green", "blue"]
for (label, data), c in zip(methods.items(), colors):
    plt.plot(data["t"], data["x"], label=label, color=c, alpha=0.85)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.title("Lorenz System: Comparison of x(t) Across Methods")
plt.grid(True)
plt.tight_layout()
plt.show(block=True)

# Error evolution (|x - x_ref|)
plt.figure(figsize=(9, 5))
for (label, vals), c in zip(metrics.items(), colors):
    err = np.abs(vals["diff"][:, 0])
    plt.plot(t_ref, err, label=label, color=c, alpha=0.9)
plt.xlabel("t")
plt.ylabel("|x - x_ref|")
plt.yscale("log")
plt.legend()
plt.title("Error Evolution in x(t) (log scale)")
plt.grid(True, which="both")
plt.tight_layout()
plt.show(block=True)

# RMSE and MAE comparison bars
labels = list(metrics.keys())
rmse_vals = [metrics[m]["rmse"] for m in labels]
mae_vals = [metrics[m]["mae"] for m in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width / 2, rmse_vals, width, label="RMSE")
bars2 = ax.bar(x + width / 2, mae_vals, width, label="MAE")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("Error")
ax.set_title("RMSE and MAE Comparison Across Methods")
ax.legend()
plt.tight_layout()
plt.show(block=True)

# Improvement percentage bar chart
labels_improv = [m for m in metrics.keys() if m != "ABM2 (Euler)"]
rmse_improv = [metrics[m]["rmse_improv"] for m in labels_improv]
mae_improv = [metrics[m]["mae_improv"] for m in labels_improv]

x = np.arange(len(labels_improv))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
bars1 = ax.bar(x - width / 2, rmse_improv, width, label="RMSE Δ%")
bars2 = ax.bar(x + width / 2, mae_improv, width, label="MAE Δ%")
ax.set_xticks(x)
ax.set_xticklabels(labels_improv, rotation=10)
ax.set_ylabel("Improvement (%)")
ax.set_title("Relative Improvement over ABM2 (Euler)")
ax.legend()
plt.tight_layout()
plt.show(block=True)
