"""
Lorenz System - Genetic Optimization of Predictor-Corrector Step Size
---------------------------------------------------------------------
This experiment applies a Genetic Algorithm (GA) to optimize the integration
step size (h) for the two-step Adams-Bashforth-Moulton (ABM2) predictor-corrector
method when solving the Lorenz system.

The GA evolves the step size that yields a stable and accurate trajectory.
The high-accuracy RK45 baseline is used internally only to guide convergence,
but no comparison is performed at this stage. The resulting trajectory will
later be compared to the fixed-step ABM2 and RK45 baselines in the final analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Lorenz parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0


def lorenz(t, state):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])


def abm2_euler(f, t0, t_end, y0, h):
    n_steps = int((t_end - t0) / h)
    t = np.linspace(t0, t_end, n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0
    f0 = f(t[0], y[0])
    y[1] = y[0] + h * f0
    f_prev, f_curr = f0, f(t[1], y[1])
    for i in range(1, n_steps):
        y_pred = y[i] + h * 0.5 * (3 * f_curr - f_prev)
        f_pred = f(t[i + 1], y_pred)
        y[i + 1] = y[i] + h * 0.5 * (f_pred + f_curr)
        f_prev, f_curr = f_curr, f_pred
    return t, y


# Load internal baseline (used only for GA guidance)
baseline = np.load("results/lorenz_baseline.npz")
t_ref, x_ref, y_ref, z_ref = baseline["t"], baseline["x"], baseline["y"], baseline["z"]
ref_traj = np.vstack((x_ref, y_ref, z_ref)).T


# Fitness function (evaluated internally)
def fitness(h):
    if h <= 0.0005 or h >= 0.02:
        return 1e6
    try:
        t, y = abm2_euler(lorenz, 0.0, 50.0, np.array([1.0, 1.0, 1.0]), h)
        if np.any(np.isnan(y)) or np.any(np.abs(y) > 1e3):
            return 1e5
        y_interp = np.array([np.interp(t_ref, t, y[:, i]) for i in range(3)]).T
        mse = np.mean((y_interp - ref_traj) ** 2)
        return mse if np.isfinite(mse) else 1e5
    except Exception:
        return 1e5


# GA parameters
pop_size = 12
generations = 20
mutation_rate = 0.25
h_bounds = (0.0005, 0.015)

population = np.random.uniform(h_bounds[0], h_bounds[1], pop_size)
best_history = []

for gen in range(generations):
    fitness_scores = np.array([fitness(h) for h in population])
    best_idx = np.argmin(fitness_scores)
    best_h = population[best_idx]
    best_score = fitness_scores[best_idx]
    best_history.append(best_score)
    print(
        f"Gen {gen + 1:02d} | best h = {best_h:.5f} | internal MSE = {best_score:.4e}"
    )

    selected = population[np.argsort(fitness_scores)[: pop_size // 2]]
    offspring = []
    for _ in range(pop_size - len(selected)):
        parents = np.random.choice(selected, 2, replace=False)
        alpha = np.random.rand()
        child = alpha * parents[0] + (1 - alpha) * parents[1]
        offspring.append(child)
    offspring = np.array(offspring)

    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            offspring[i] += np.random.uniform(-0.001, 0.001)
            offspring[i] = np.clip(offspring[i], *h_bounds)

    population = np.concatenate((selected, offspring))

# Final integration using best step size
t_opt, y_opt = abm2_euler(lorenz, 0.0, 50.0, np.array([1.0, 1.0, 1.0]), best_h)

print(f"\nGA complete: best step size h = {best_h:.5f}")


# Save results
os.makedirs("results", exist_ok=True)
np.savez(
    "results/lorenz_pc_ga_optimized.npz",
    t=t_opt,
    x=y_opt[:, 0],
    y=y_opt[:, 1],
    z=y_opt[:, 2],
    h=best_h,
)

# --- GA Convergence Plot ---
plt.figure(figsize=(7, 4))
plt.plot(best_history, color="teal")
plt.yscale("log")
plt.xlabel("Generacija")
plt.ylabel("MSE (log)")
# plt.title("GA Convergence for Step Size Optimization")
plt.tight_layout()
plt.grid(True)
plt.savefig("results/images/convergence_ga_optimized.png", dpi=220)
plt.show(block=True)

# --- 3D Trajectory ---
fig1 = plt.figure(figsize=(7, 6))
ax1 = fig1.add_subplot(111, projection="3d")
ax1.plot(y_opt[:, 0], y_opt[:, 1], y_opt[:, 2], lw=0.6, color="forestgreen")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
# ax1.set_title(f"Lorenz Attractor (GA-Optimized ABM2, h={best_h:.4f})")
plt.tight_layout()
plt.savefig("results/images/lorenz_ga_optimized.png", dpi=220)
plt.show(block=True)

# --- Time Series ---
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.plot(t_opt, y_opt[:, 0], label="x(t)")
ax2.plot(t_opt, y_opt[:, 1], label="y(t)")
ax2.plot(t_opt, y_opt[:, 2], label="z(t)")
ax2.set_xlabel("t")
ax2.set_ylabel("$u(t)$")
ax2.legend()
# ax2.set_title("Lorenz System State Evolution (GA-Optimized ABM2)")
plt.tight_layout()
plt.grid(True)
plt.savefig("results/images/lorenz_ga_optimized_timeseries.png", dpi=220)
plt.show(block=True)
