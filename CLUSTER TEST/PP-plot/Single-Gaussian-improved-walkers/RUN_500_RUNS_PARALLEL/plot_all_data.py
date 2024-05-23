import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import shutil

N_obs = 500

for ii in range(1, 101):
    if os.path.isdir(f'Child-Gaussian/run_{ii}'):
        shutil.rmtree(f'Child-Gaussian/run_{ii}')

    os.mkdir(f'Child-Gaussian/run_{ii}')
    print("Directory created") 
    hf = h5py.File(f'data/run_{ii}.h5')
    data_group = hf.get('data')
    data_child = np.array(data_group.get('data'))
    data_parent = np.array(data_group.get('hyper_data'))
    data_error = np.array(data_group.get('error_on_data'))

    injection_group = hf.get('injections')
    injection_mu_p = np.array(injection_group.get('mu_p'))
    injection_sigma_p = np.array(injection_group.get('sigma_p'))

    for i in range(N_obs):
        plt.hist(data_child[i], fill=False, histtype='step')
        plt.vlines(data_parent[i], 0, 50, linestyles='-.', color='green', label='true')
        plt.vlines(data_parent[i] + data_error[i], 0, 50, linestyles='--', color='red', label='error added')
        plt.ylim(0,50)
        plt.legend()
        plt.savefig(f'Child-Gaussian/run_{ii}/child_{i}.png',dpi=100)
        plt.close()

    plt.hist(data_parent, fill=False, histtype='step', color='black',linewidth=3)
    plt.vlines(injection_mu_p, 0, 200, color='black',linewidth=3)
    plt.title(f'$N({injection_mu_p},{injection_sigma_p})$')
    plt.ylim(0,200)
    plt.savefig(f'Parent-Gaussian/run_{ii}.png',dpi=100)
    plt.close()

    hf.close()