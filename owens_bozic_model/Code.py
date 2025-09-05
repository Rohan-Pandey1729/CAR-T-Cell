import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Defining the model

#if you start with 0 CART and CHemo cells, and don't add any, then the Indogenous cells is at a low level, and the Tumor Cells grows.
def model(y, t, params):
    T, E, C, M = y
    # a, b, dE, s, l, mE, g, jE, qE, k, KE, KC, mC, jC, qC, dC, gammaM, KT = params
    a, b, dE, dC, g, jE, jC, K, k, l, mE, mC, qE, qC, s, KT, KE, KC, gamma = params
    epsilon = 1e-6

    C = max(C, epsilon)
    E = max(E, epsilon)
    
    DE = dE * (E / T)**l / (s + (E / T)**l) * T
    DC = dC * (C / T)**l / (s + (C / T)**l) * T
    
    dT_dt = a * T * (1 - b * T) - DE - DC - KT * (1 - np.exp(-M)) * T
    dE_dt = g - mE * E - jE * np.log((E + C) / K) * (DE**2) / (k + DE**2) * E  - qE * E * T - KE * (1 - np.exp(-M)) * E
    dC_dt = - mC * C - jC * np.log((E + C) / K) * (DC**2) / (k + DC**2) * C - qC * C * T - KC * (1 - np.exp(-M)) * C
    dM_dt = - gamma * M
    
    return [dT_dt, dE_dt, dC_dt, dM_dt]

# Function to simulate the model

''' 
For the v_c, v_m you want to consider it as another injection time
Now we're injecting CAR-T cells
Try to set it up to inject 1 CAR-T after doses of Chemo

Future plans: Multiple doses of CAR-T cells
'''

def simulate(model, y0, params, injection_times_M, dosage_M, t_initial, t_final, time_step):
    t = np.arange(t_initial, injection_times_M[0] + time_step, time_step)
    solution = odeint(model, y0, t, args=(params,))
    yt = solution[-1] + [0, 0, 0, dosage_M]

    for i in range(1, len(injection_times_M)):
        temp_time = np.arange(injection_times_M[i-1], injection_times_M[i] + time_step, time_step)
        t = np.concatenate((t, temp_time))
        sol = odeint(model, yt, temp_time, args=(params,))
        yt = sol[-1] + [0, 0, 0, dosage_M]
        solution = np.vstack((solution, sol))

    temp_time = np.arange(injection_times_M[-1], t_final + time_step, time_step)
    t = np.concatenate((t, temp_time))
    sol = odeint(model, yt, temp_time, args=(params,))
    solution = np.vstack((solution, sol))

    return t, solution

# second param is b, not 1/b

params_patient_1 = [2.55e-1, 5e-13, 2.03, 2.25, 1.4e3, 1.1e-2, 2.42e-1, 1.65e9, 2.019e5, 1.395, 7e-3, 2.93e-2, 3.42e-11, 3.0e-11, 3.05e-1, 7e-1, 6e-1, 6e-1, 9e-1]

# different ordering than above
# params_1 = [2.55e-1, 5e-13, 2.03, 3.0e-1, 1.395, 7e-3, 1.43, 1.1e-2, 3.42e-10, 1.65e9, 
        #   2.019e5, 2.019e5, 2.93e-2, 1.42e-1, 3.42e-10, 2.25, 0.1, 3.05e-1]



params = params_patient_1

y0_patient_1 = [5e9, 4e5, 0, 0]
y0 = y0_patient_1

# Remember that y0 is all your T, E, C, M, try playing around and add a little bit of Chemo/Treatment

# injection_times_M = [2, 4, 6]
injection_times_M = [0, 1, 2]
dosage_M = 5


t_no_treatment, solution_no_treatment = simulate(model, y0, params, injection_times_M, dosage_M, t_initial=0, t_final=30, time_step=0.01)



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
plt.xlim(0, 50)
plt.ylim(1, 1e15)

plt.xlabel('Time (days)')
plt.ylabel('Cells')
plt.title('Cancer Cell Population Dynamics with Treatment')
plt.legend(loc='best')
plt.grid(True)
plt.show()
