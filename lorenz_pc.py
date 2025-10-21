"""
Lorenz System - Predictor-Corrector (Adams-Bashforth-Moulton, Euler Bootstrap)
------------------------------------------------------------------------------
This implementation integrates the Lorenz system using the two-step
Adams-Bashforth-Moulton (ABM2) predictor-corrector method. A single
Euler step is used for initialization, providing the second data point
required by the multistep scheme. The ABM2 method combines an explicit
predictor (Adams-Bashforth) and an implicit corrector (Adams-Moulton)
to achieve second-order accuracy while maintaining moderate computational
cost. The resulting trajectory is saved for quantitative comparison with
the high-accuracy Runge-Kutta (RK45) baseline solution.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

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


def abm2_euler(f, t0, t_end, y0, h):
    n_steps = int((t_end - t0) / h)
    t = np.linspace(t0, t_end, n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0

    # Euler bootstrap - generally RK4 is preferred for better accuracy
    f0 = f(t[0], y[0])
    y[1] = y[0] + h * f0
    f_prev, f_curr = f0, f(t[1], y[1])

    for i in range(1, n_steps):
        y_pred = y[i] + h * 0.5 * (3 * f_curr - f_prev)
        f_pred = f(t[i + 1], y_pred)
        y[i + 1] = y[i] + h * 0.5 * (f_pred + f_curr)
        f_prev, f_curr = f_curr, f_pred
        if i % 1000 == 0:
            print(f"Step {i}/{n_steps}")

    return t, y


# Integration setup
t0, t_end = 0.0, 50.0
h = 0.01
y0 = np.array([1.0, 1.0, 1.0])

print("Computing Lorenz system using ABM2 with Euler initialization...")
t, y = abm2_euler(lorenz, t0, t_end, y0, h)
print("Integration complete.")

os.makedirs("results", exist_ok=True)
np.savez("results/lorenz_pc_abm2_euler.npz", t=t, x=y[:, 0], y=y[:, 1], z=y[:, 2])
print("Results saved to results/lorenz_pc_abm2_euler.npz")

# Visualization
fig1 = plt.figure(figsize=(7, 6))
ax1 = fig1.add_subplot(111, projection="3d")
ax1.plot(y[:, 0], y[:, 1], y[:, 2], lw=0.6, color="darkorange")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
# ax1.set_title("Lorenz Attractor (ABM Predictor–Corrector, Euler Bootstrap)")
plt.tight_layout()
plt.show(block=True)

fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.plot(t, y[:, 0], label="x(t)")
ax2.plot(t, y[:, 1], label="y(t)")
ax2.plot(t, y[:, 2], label="z(t)")
ax2.set_xlabel("t")
ax2.set_ylabel("$u(t)$")
ax2.legend()
# ax2.set_title("Lorenz System State Evolution (ABM2, Euler Bootstrap)")
plt.tight_layout()
plt.grid(True)
plt.show(block=True)
