# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse as spr

font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 50}

matplotlib.rc('font', **font)
#%%
dir_base = 'ijrmms/tri3/G0.10000Daeff0.00400/'
data_all = np.empty((5, 0)).tolist()
plt.figure(figsize = (15, 10))
for i in range(1, 31):
    #dirname = f'dissolutionG01Da001/carbonate_x{i:02}/slices.txt'
    dirname = dir_base + f'n100l100r1_{i:02}/profiles.txt'
    f = open(dirname, 'r')
    data = np.loadtxt(f)
    edge_number  = np.array(data[0])
    for j, data_t in enumerate(data[1:]):
        print(j)
        #data_all[i].append(np.array(data_t)/np.array(data[1]))
        data_all[j].append(np.array((edge_number - 2 * np.array(data_t)) \
            / edge_number))

x = np.linspace(0, 100, 100)
labels = ['0.0', '1.0', '2.0', '5.0', '20.0']
colors = ['black', 'C0', 'C1', 'C2', 'C3']

plt.plot([], [], ' ', label=' ')
plt.plot([], [], ' ', label=' ')
plt.plot([], [], ' ', label=' ')
for i, data in enumerate(data_all):
    mean_profile = np.mean(data_all[i], axis=0)
    std_profile = np.std(data_all[i], axis=0)
    sem_profile = std_profile / np.sqrt(len(data_all[i]))
    err = np.std(data_all[i], axis = 0, ddof=0)
    err = np.std(data_all[i], axis = 0, ddof=1) / np.sqrt(np.size(data_all[0]))
    #plt.errorbar(x, np.average(data_all[i], axis = 0), yerr = err)
    plt.plot(x, mean_profile, linewidth = 5, color = colors[i], label = labels[i])
    plt.fill_between(x, mean_profile - std_profile, mean_profile + std_profile,
                   color=colors[i], alpha=0.2)
#plt.ylim(0, 1.05)
plt.xlabel('x', fontsize = 60, style = 'italic')
plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(tight = True)
plt.ylabel('flow focusing index', fontsize = 50)
#plt.yticks([0, 0.5, 1],['0', '0.5', '1'])
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,4,1,5,2,6,3,7]

legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="lower center", mode = "expand", ncol = 4, prop={'size': 40}, handlelength = 1, frameon=False, borderpad = 0, handletextpad = 0.4)
for legobj in legend.legend_handles:
    legobj.set_linewidth(10.0)
plt.savefig(dir_base + 'mean_std.png', bbox_inches="tight")
plt.show()
plt.close()
labels = ['0.0', '1.0', '2.0', '5.0', '20.0']
colors = ['black', 'C0', 'C1', 'C2', 'C3']
plt.figure(figsize = (15, 10))
plt.plot([], [], ' ', label=' ')
plt.plot([], [], ' ', label=' ')
plt.plot([], [], ' ', label=' ')
for i, data in enumerate(data_all):
    mean_profile = np.mean(data_all[i], axis=0)
    std_profile = np.std(data_all[i], axis=0)
    sem_profile = std_profile / np.sqrt(len(data_all[i]))
    err = np.std(data_all[i], axis = 0, ddof=0)
    err = np.std(data_all[i], axis = 0, ddof=1) / np.sqrt(np.size(data_all[0]))
    #plt.errorbar(x, np.average(data_all[i], axis = 0), yerr = err)
    plt.plot(x, mean_profile, linewidth = 5, color = colors[i], label = labels[i])
    #plt.fill_between(x, mean_profile - std_profile, mean_profile + std_profile,
    #               color=colors[i], alpha=0.2)
plt.ylim(0, 1.05)
plt.xlabel('x', fontsize = 60, style = 'italic')
plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(tight = True)
plt.ylabel('flow focusing index', fontsize = 50)
plt.yticks([0, 0.5, 1],['0', '0.5', '1'])
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,4,1,5,2,6,3,7]

legend = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="lower center", mode = "expand", ncol = 4, prop={'size': 40}, handlelength = 1, frameon=False, borderpad = 0, handletextpad = 0.4)
for legobj in legend.legend_handles:
    legobj.set_linewidth(10.0)
#plt.savefig(dirname[:-24] + dirname[13:-25] + "mean.png", bbox_inches="tight")
plt.savefig(dir_base + 'mean.png', bbox_inches="tight")
plt.show()
plt.close()
# %%
