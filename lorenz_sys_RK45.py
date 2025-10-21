"""
Lorenz System - Baseline Simulation
-----------------------------------
This script computes and visualizes a reference (baseline) solution
of the Lorenz system using the Runge-Kutta 4/5 (RK45) method via SciPy's
`solve_ivp`. The output is saved for later comparison against
Predictor-Corrector and GA-optimized Predictor-Corrector methods.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# System parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0


def lorenz(t, state):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


# Initial conditions and integration setup
x0, y0, z0 = (1.0, 1.0, 1.0)
t_span = (0, 50)
t_eval = np.linspace(*t_span, 10000)

# Numerical solution using RK45
solution = solve_ivp(lorenz, t_span, [x0, y0, z0], t_eval=t_eval, method="RK45")

if not solution.success:
    raise RuntimeError("Integration failed.")

os.makedirs("results", exist_ok=True)
np.savez(
    "results/lorenz_baseline.npz",
    t=solution.t,
    x=solution.y[0],
    y=solution.y[1],
    z=solution.y[2],
)
print("Baseline solution saved to results/lorenz_baseline.npz")

# load if necessary
# base = np.load("results/lorenz_baseline.npz")

# 3d trajectory
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection="3d")
ax1.plot(solution.y[0], solution.y[1], solution.y[2], lw=0.6, color="purple")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
# ax1.set_title("Lorenz Attractor (Baseline RK45)")
plt.tight_layout()
plt.savefig("results/images/lorenz_baseline_rk45.png", dpi=300)
plt.show()

# time series
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.plot(solution.t, solution.y[0], label="x(t)")
ax2.plot(solution.t, solution.y[1], label="y(t)")
ax2.plot(solution.t, solution.y[2], label="z(t)")
ax2.set_xlabel("t")
ax2.set_ylabel("$u(t)$")
ax2.legend()
# ax2.set_title("Lorenz System State Evolution")
plt.tight_layout()
plt.grid(True)
plt.savefig("results/images/lorenz_baseline_rk45_timeseries.png", dpi=220)
plt.show()
