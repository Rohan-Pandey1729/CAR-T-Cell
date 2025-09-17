# car_t_fixed.py
# Nondimensional CAR-T model aligned with your PlayingWithModels structure
# States: x = T/K, y = C/C50, z = M/C50; time: s = r * t

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------- Replace these dimensional parameters with your real values ----------
params = {
    'r': 0.1,        # tumor growth rate [1/time]
    'K': 1e9,        # tumor carrying capacity [cells]
    'kappa': 0.05,   # killing rate [1/time]
    'C50': 1e6,      # C50 [cells]
    'p': 0.2,        # effector proliferation [1/time]
    'ell': 0.01,     # effector -> memory [1/time]
    'delta_C': 0.01, # effector death [1/time]
    'delta_M': 0.001,# memory death [1/time]
    'q': 0.05,       # exhaustion weight [1/time]
    's': 0.1,        # exhaustion shape (dimensionless in model)
    'b': 0.02,       # memory -> effector reseed [1/time]
    'T50_1': 1e7,    # T50 for effector proliferation [cells]
    'T50_2': 5e6,    # T50 for memory reseed coupling [cells]
    'h1': 2.0,       # Hill exponent for proliferation
    'h2': 1.0        # Hill exponent for reseed coupling
}
# ------------------------------------------------------------------------------

# Compute nondimensional groups
r = params['r']
K = params['K']
C50 = params['C50']

alpha = params['kappa'] / r               # tumor kill group
beta = params['p'] / r                    # effector proliferation
lam = params['ell'] / r                   # effector->memory
delta_c = params['delta_C'] / r           # effector death
delta_m = params['delta_M'] / r           # memory death
phi = params['q'] / r                     # exhaustion weight
gamma = params['b'] / r                   # memory->effector reseed
theta1 = params['T50_1'] / K              # T50 for F1
theta2 = params['T50_2'] / K              # T50 for F2
mu = params['s'] * C50 / K                # exhaustion shape -> x/(x+mu y)
h1 = params['h1']
h2 = params['h2']

dimless = dict(alpha=alpha, beta=beta, lam=lam, delta_c=delta_c,
               delta_m=delta_m, phi=phi, gamma=gamma,
               theta1=theta1, theta2=theta2, mu=mu, h1=h1, h2=h2)

print("Dimensionless parameters:")
for k, v in dimless.items():
    print(f"  {k} = {v}")

# --- helpers (Hill gates) ---
def hill(x_val, theta, h):
    xx = max(x_val, 0.0)
    num = xx**h
    den = num + theta**h
    return num/den if den != 0.0 else 0.0

# RHS of nondimensional ODEs (x,y,z) vs s
def rhs(s, Y):
    x, y, z = Y
    # tumor
    dx = x*(1.0 - x) - alpha * (y/(1.0 + y)) * x
    # gates
    F1 = hill(x, theta1, h1)  # proliferation gate
    F2 = hill(x, theta2, h2)  # reseed coupling gate
    Exhaust = (x/(x + mu*y)) if (x + mu*y) != 0.0 else 0.0
    # effector
    dy = (beta * y * F1          # + proliferation
          + gamma * z * F2       # + reseed from memory
          - (lam + delta_c) * y  # - transfer to memory - death
          - phi * y * Exhaust)   # - exhaustion
    # memory
    dz = lam * y - gamma * z * F2 - delta_m * z
    return [dx, dy, dz]

# Initial conditions (NONDIM): x(0)=T0/K, y(0)=C0/C50, z(0)=M0/C50
Y0 = [0.1, 0.01, 0.0]

# Solve over dimensionless time s ( = r * t )
t_span = (0.0, 100.0)
sol = solve_ivp(rhs, t_span, Y0, dense_output=True, atol=1e-8, rtol=1e-6)

# Evaluate & report
ts = np.linspace(t_span[0], t_span[1], 400)
Ys = sol.sol(ts)
x_ts, y_ts, z_ts = Ys
print("\nFinal state (dimensionless):")
print(f"  x({t_span[1]}) = {x_ts[-1]}")
print(f"  y({t_span[1]}) = {y_ts[-1]}")
print(f"  z({t_span[1]}) = {z_ts[-1]}")

# Plot style similar to your notebook; toggle semilogy as needed
USE_SEMILOGY = False

plt.figure(figsize=(8,4.5))
if USE_SEMILOGY:
    plt.semilogy(ts, np.maximum(x_ts, 1e-12), label='x = T/K')
    plt.semilogy(ts, np.maximum(y_ts, 1e-12), label='y = C/C50')
    plt.semilogy(ts, np.maximum(z_ts, 1e-12), label='z = M/C50')
else:
    plt.plot(ts, x_ts, label='x = T/K')
    plt.plot(ts, y_ts, label='y = C/C50')
    plt.plot(ts, z_ts, label='z = M/C50')

plt.xlabel('dimensionless time s = r t')
plt.legend()
plt.tight_layout()
plt.show()
