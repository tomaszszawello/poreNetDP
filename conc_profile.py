#%%
import numpy as np
import matplotlib.pyplot as plt

dirname = 'fracture_diffusion/Pe0.00Da100000.00/4/'
dirname = 'fracture_diffusion/Pe0.00Da10000000.00/0/'
dirname = 'fracture_diffusion/Pe0.00Da1000000.00/1/'
M = 200
Da = 10 # pi d0 l0 k / q0
Pe = 0.05 # 4 q0 l0 / pi d0^2 D2
fD = 1 # D1 / D2
r1 = 1 #np.sqrt(0.3 / 1e-6) # d1 / d0
r2 = 1 # d2 / d0

Pe1 = Pe / fD / r1 ** 2
Lam = np.sqrt(Pe * Da / r2)
a = r2 ** (3 / 2) * np.sqrt(Da / Pe)
conc_data = np.loadtxt(dirname + 'concentration_y0.txt')
front_pos = np.loadtxt(dirname + 'front_pos.txt')
slice_times = np.loadtxt(dirname + 'slice_times.txt')
slice_time = 5
eps = 1e-10
x = np.arange(M, step = M / len(conc_data[0]))
front_x = front_pos[np.argmin(np.abs(front_pos[:,0] - slice_times[slice_time])), 2] + eps
print(front_x)
#c_l = 1 / (1 + a * (1 - np.exp(-Pe1 * front_x)))

c_l_diff = 1 / (1 + front_x * a / (fD * r1 ** 2))
c_diff = (x <= front_x) * (1 + (c_l_diff - 1) * x / front_x) + (x > front_x) * c_l_diff * np.exp(-Lam * (x - front_x))
c_l_adv = 1 / (1 + a * (1 - np.exp(-Pe1 * front_x)))
c_adv = (x <= front_x) * (1 + (c_l_adv - 1) * (np.exp(Pe1 * x) - 1) / (np.exp(Pe1 * front_x) - 1)) + (x > front_x) * c_l_adv * np.exp(-Lam * (x - front_x))
plt.plot(x, c_diff)
plt.plot(x, c_adv)
plt.plot(x, conc_data[slice_time])
plt.show()
# %%
