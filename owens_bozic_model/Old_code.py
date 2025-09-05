import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Defining the model
def model(y, t, params):  # Added 't' as odeint expects this
    T, E, C, M = y
    a, b, dE, s, l, mE, g, jE, qE, k, KE, KC, mC, jC, qC, dC, gammaM, KT = params
    epsilon = 1e-6
    
    T_safe = max(T, epsilon)
    E_safe = max(E, epsilon)
    C_safe = max(C, epsilon)
    
    DE = dE * (E_safe / T_safe)**l / (s + (E_safe / T_safe)**l) * T_safe
    DC = dC * (C_safe / T_safe)**l / (s + (C_safe / T_safe)**l) * T_safe
    
    dT_dt = a * T_safe * (1 - b * T_safe) - DE - DC - KT * (1 - np.exp(-M)) * T_safe
    dE_dt = g - mE * E_safe - jE * E_safe * np.log((E_safe + C_safe) / k) - (DE**2 * E_safe) / (k + DE**2) - qE * E_safe * T_safe - KE * (1 - np.exp(-M)) * E_safe
    dC_dt = - mC * C_safe - jC * C_safe * np.log((E_safe + C_safe) / k) - (DC**2 * C_safe) / (k + DC**2) - qC * C_safe * T_safe - KC * (1 - np.exp(-M)) * C_safe
    dM_dt = - gammaM * M
    
    return [dT_dt, dE_dt, dC_dt, dM_dt]

# Function to simulate the model
def simulate(model, y0, params, injection_times_M, dosage_M, t_initial, t_final, time_step):
    t = np.arange(t_initial, injection_times_M[0] + time_step, time_step)
    solution = odeint(model, y0, t, args=(params,))
    
    yt = solution[-1]
    yt[3] += dosage_M  # Adding the dosage to the M component only

    for i in range(1, len(injection_times_M)):
        temp_time = np.arange(injection_times_M[i-1], injection_times_M[i] + time_step, time_step)
        t = np.concatenate((t, temp_time))
        sol = odeint(model, yt, temp_time, args=(params,))
        yt = sol[-1]
        yt[3] += dosage_M  # Adding the dosage to the M component at each injection time
        solution = np.vstack((solution, sol))

    temp_time = np.arange(injection_times_M[-1], t_final + time_step, time_step)
    t = np.concatenate((t, temp_time))
    sol = odeint(model, yt, temp_time, args=(params,))
    solution = np.vstack((solution, sol))

    return t, solution

# Parameters
params = [2.55e-1, 5e-13, 2.03, 3.0e-1, 1.395, 7e-3, 1.43, 1.1e-2, 3.42e-10, 1.65e9, 
          2.019e5, 2.019e5, 2.93e-2, 1.42e-1, 3.42e-10, 2.25, 0.1, 3.05e-1]

# Initial conditions: Tumor cells (T), Effector cells (E), CAR-T cells (C), and Chemotherapy (M)
y0 = [1e6, 1e5, 0, 0]

# Injection times and dosage
injection_times_M = [2, 4, 6]
dosage_M = 5  # Adjust dosage accordingly

# Simulation without treatment
t_no_treatment, solution_no_treatment = simulate(model, y0, params, injection_times_M, dosage_M, t_initial=0, t_final=30, time_step=0.01)

# Extracting the solution components
T = solution_no_treatment[:, 0]
E = solution_no_treatment[:, 1]
C = solution_no_treatment[:, 2]
M = solution_no_treatment[:, 3]

# Plotting the results
plt.figure(figsize=(14, 10))

plt.plot(t_no_treatment, T, label='Tumor Cells T(t)', color='red', linewidth=2)
plt.plot(t_no_treatment, E, label='Effector Cells E(t)', color='blue', linewidth=2)
plt.plot(t_no_treatment, C, label='CAR-T Cells C(t)', color='green', linewidth=2)
plt.plot(t_no_treatment, M, label='Chemotherapy M(t)', color='purple', linewidth=2)

plt.yscale('log')
plt.xlim(0, 30)  # Adjust this range as necessary
plt.ylim(1e2, 1e10)  # Adjust this range to better visualize

plt.xlabel('Time (days)')
plt.ylabel('Cells')
plt.title('Cancer Cell Population Dynamics with Treatment')
plt.legend(loc='best')
plt.grid(True)
plt.show()
