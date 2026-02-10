
import h5py
import numpy as np 

dirname_base = f'check_new_Da/G0.10000Daeff0.00020/'
name = "G0_10_Daeff0_0002.hdf5"
with h5py.File("adv_t_" + name, "w") as f5file:
    for i in range(1, 31):
        dirname = dirname_base + f'carbonate_x{i:02}/'
        f = open(dirname + 'tracks_num.txt', 'r')
        data = np.loadtxt(f)
        cp_subgroup = f5file.create_group(f'carbonate_x{i:02}')

        dataset_names = ['0.00', '0.10', '0.20', '0.50', '1.00']
        for j in range(0, 5):
            cp_subgroup.create_dataset(f"t_{dataset_names[j]}", data=data[j], dtype='float64')
with h5py.File("conc_" + name, "w") as f5file:
    for i in range(1, 31):
        dirname = dirname_base + f'carbonate_x{i:02}/'
        f = open(dirname + 'c_tracks_num.txt', 'r')
        data = np.loadtxt(f)
        cp_subgroup = f5file.create_group(f'carbonate_x{i:02}')

        dataset_names = ['0.00', '0.10', '0.20', '0.50', '1.00']
        for j in range(0, 5):
            cp_subgroup.create_dataset(f"t_{dataset_names[j]}", data=data[j], dtype='float64')
with h5py.File("pl_" + name, "w") as f5file:
    for i in range(1, 31):
        dirname = dirname_base + f'carbonate_x{i:02}/'
        f = open(dirname + 'pl_tracks_num.txt', 'r')
        data = np.loadtxt(f)
        cp_subgroup = f5file.create_group(f'carbonate_x{i:02}')

        dataset_names = ['0.00', '0.10', '0.20', '0.50', '1.00']
        for j in range(0, 5):
            cp_subgroup.create_dataset(f"t_{dataset_names[j]}", data=data[j], dtype='float64')

