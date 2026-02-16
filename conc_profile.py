#%%
import numpy as np
import matplotlib.pyplot as plt

dirname = 'fracture_diffusion/Pe0.01Da10000.00/9/'
N_an  = 100
Da = 1 # k l0^2 / D2
Pe = 0.4 # u0 l0 / D1
f_D = 1000 # D1 / D2
conc_data = np.loadtxt(dirname + 'concentration_y0.txt')
front_pos = np.loadtxt(dirname + 'front_pos.txt')
slice_times = np.loadtxt(dirname + 'slice_times.txt')
slice_time = 10
eps = 1e-10
x = np.linspace(0, len(conc_data[slice_time]), N_an)
front_x = front_pos[np.argmin(np.abs(front_pos[:,0] - slice_times[slice_time])), 2] + eps
print(front_x)


c_l_diff = 1 / (1 + front_x * np.sqrt(Da) / f_D)
c_diff = (x <= front_x) * (1 + (c_l_diff - 1) * x / front_x) + (x > front_x) * c_l_diff * np.exp(-np.sqrt(Da) * (x - front_x))
c_l_adv = c_l_adv = 1.0 / (1.0 + (np.sqrt(Da)/(Pe*f_D)) * (1.0 - np.exp(-Pe*front_x)))
c_adv = (x <= front_x) * (1 + (c_l_adv - 1) * (np.exp(Pe * x) - 1) / (np.exp(Pe * front_x) - 1)) + (x > front_x) * c_l_adv * np.exp(-np.sqrt(Da) * (x - front_x))
plt.plot(x, c_diff)
plt.plot(x, c_adv)
plt.plot(conc_data[slice_time])
plt.show()

# %%
