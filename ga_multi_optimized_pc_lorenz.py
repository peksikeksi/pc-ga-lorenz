"""
Lorenz System - GA Optimization of Predictor-Corrector Coefficients
-------------------------------------------------------------------
This script evolves both the step size and the predictor/corrector
coefficients of the two-step Adams-Bashforth-Moulton method for the
Lorenz system. The Genetic Algorithm minimizes the mean squared error
between the GA-tuned ABM2 trajectory and the high-accuracy RK45 baseline.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp

# Lorenz system parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0


def lorenz(t, state):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])


# Generalized ABM2 with adjustable coefficients
def abm2_general(f, t0, t_end, y0, h, a1, a2, b1, b2):
    n_steps = int((t_end - t0) / h)
    t = np.linspace(t0, t_end, n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0

    # Euler bootstrap
    f0 = f(t[0], y[0])
    y[1] = y[0] + h * f0
    f_prev, f_curr = f0, f(t[1], y[1])

    for i in range(1, n_steps):
        y_pred = y[i] + h * (a1 * f_curr + a2 * f_prev)
        f_pred = f(t[i + 1], y_pred)
        y[i + 1] = y[i] + h * (b1 * f_pred + b2 * f_curr)
        f_prev, f_curr = f_curr, f_pred

        if np.any(np.isnan(y[i + 1])) or np.any(np.abs(y[i + 1]) > 1e3):
            return t[: i + 1], y[: i + 1]

    return t, y


# Load RK45 baseline
baseline = np.load("results/lorenz_baseline.npz")
t_ref, x_ref, y_ref, z_ref = baseline["t"], baseline["x"], baseline["y"], baseline["z"]
ref_traj = np.vstack((x_ref, y_ref, z_ref)).T


# Fitness function
def fitness(params):
    h, a1, a2, b1, b2 = params
    if not (0.0005 <= h <= 0.02 and all(0 <= x <= 2 for x in [a1, a2, b1, b2])):
        return 1e6
    try:
        t, y = abm2_general(
            lorenz, 0.0, 30.0, np.array([1.0, 1.0, 1.0]), h, a1, a2, b1, b2
        )
        if len(t) < 1000 or np.any(np.isnan(y)) or np.any(np.abs(y) > 1e3):
            return 1e5
        y_interp = np.array([np.interp(t_ref, t, y[:, i]) for i in range(3)]).T
        mse = np.mean((y_interp - ref_traj) ** 2)
        if not np.isfinite(mse):
            mse = 1e5
        return mse
    except Exception:
        return 1e5


# GA setup
pop_size = 30
generations = 40
mutation_rate = 0.3

# Random initial population
population = np.array(
    [
        [
            np.random.uniform(0.001, 0.01),  # h
            np.random.uniform(1.0, 1.6),  # a1
            np.random.uniform(-0.2, 0.6),  # a2
            np.random.uniform(0.4, 0.7),  # b1
            np.random.uniform(0.3, 0.6),
        ]  # b2
        for _ in range(pop_size)
    ]
)

best_history = []

for gen in range(generations):
    fitness_scores = np.array([fitness(ind) for ind in population])
    best_idx = np.argmin(fitness_scores)
    best_params = population[best_idx]
    best_score = fitness_scores[best_idx]
    best_history.append(best_score)

    print(
        f"Gen {gen + 1:02d} | best h={best_params[0]:.5f} | a1={best_params[1]:.3f}, a2={best_params[2]:.3f}, b1={best_params[3]:.3f}, b2={best_params[4]:.3f} | MSE={best_score:.4e}"
    )

    selected = population[np.argsort(fitness_scores)[: pop_size // 2]]

    # Crossover
    offspring = []
    for _ in range(pop_size - len(selected)):
        parents = np.random.choice(len(selected), 2, replace=False)
        alpha = np.random.rand()
        child = alpha * selected[parents[0]] + (1 - alpha) * selected[parents[1]]
        offspring.append(child)
    offspring = np.array(offspring)

    # Mutation
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            offspring[i] += np.random.normal(0, 0.05, size=5)
            offspring[i][0] = np.clip(offspring[i][0], 0.0005, 0.02)
            offspring[i][1:] = np.clip(offspring[i][1:], 0.0, 2.0)

    population = np.concatenate((selected, offspring))

# Final evaluation
best_fitness = fitness(best_params)
t_opt, y_opt = abm2_general(lorenz, 0.0, 50.0, np.array([1.0, 1.0, 1.0]), *best_params)
print(f"\nGA optimization complete:")
print(
    f"Best parameters: h={best_params[0]:.5f}, a1={best_params[1]:.3f}, a2={best_params[2]:.3f}, b1={best_params[3]:.3f}, b2={best_params[4]:.3f}"
)
print(f"Final MSE={best_fitness:.4e}")

os.makedirs("results", exist_ok=True)
np.savez(
    "results/lorenz_pc_ga_multi.npz",
    t=t_opt,
    x=y_opt[:, 0],
    y=y_opt[:, 1],
    z=y_opt[:, 2],
    params=best_params,
)

# --- GA Convergence Plot ---
plt.figure(figsize=(7, 4))
plt.plot(
    best_history,
)
plt.yscale("log")
plt.xlabel("Generacija")
plt.ylabel("MSE (log scale)")
# plt.title("GA Convergence of Predictor-Corrector Parameters")
plt.tight_layout()
plt.grid(True)
plt.savefig("results/images/convergence_multi_ga_optimized.png", dpi=220)
plt.show(block=True)

# --- 3D Trajectory ---
fig1 = plt.figure(figsize=(7, 6))
ax1 = fig1.add_subplot(111, projection="3d")
ax1.plot(y_opt[:, 0], y_opt[:, 1], y_opt[:, 2], lw=0.6, color="green")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
# ax1.set_title("Lorenz Attractor (GA-Optimized ABM2 Coefficients)")
plt.tight_layout()
plt.savefig("results/images/lorenz_multi_ga_optimized.png", dpi=220)
plt.show(block=True)

# --- Time-Series ---
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.plot(t_opt, y_opt[:, 0], label="x(t)")
ax2.plot(t_opt, y_opt[:, 1], label="y(t)")
ax2.plot(t_opt, y_opt[:, 2], label="z(t)")
ax2.set_xlabel("t")
ax2.set_ylabel("$u(t)$")
ax2.legend()
# ax2.set_title("Lorenz System State Evolution (GA-Optimized ABM2 Coefficients)")
plt.tight_layout()
plt.grid(True)
plt.savefig("results/images/lorenz_multi_ga_optimized_timeseries.png", dpi=220)
plt.show(block=True)
