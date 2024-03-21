from multiprocessing import Pool, cpu_count

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from starcat import (Isoc, Parsec, IMF,
                     BinMRD,
                     Individual, SynStars,
                     lnlike, Hist2Point4CMD, Gaussian, Hist2Hist4CMD)

# load observation data for photometric error
file_path = '/Users/sara/PycharmProjects/starcat/data/Almeida23/membership_edr3/Melotte_22_data_stars.npy'
obs_sample = np.load(file_path)
selected_columns = ['Gmag', 'BPmag', 'RPmag', 'e_Gmag', 'e_BPmag', 'e_RPmag']
Pobservation = pd.DataFrame(obs_sample[selected_columns])

mag_cut = 18.
Pobservation = Pobservation[(Pobservation['BPmag'] - Pobservation['RPmag']) > -0.2]
Pobservation = Pobservation[Pobservation['Gmag'] < mag_cut]
c = Pobservation['BPmag'] - Pobservation['RPmag']
sigma_c = np.sqrt((Pobservation['e_BPmag']) ** 2 + (Pobservation['e_RPmag']) ** 2)
Pobservation = Pobservation[sigma_c < np.std(sigma_c) * 3]

print(max(Pobservation['Gmag']), max(Pobservation['BPmag']), max(Pobservation['RPmag']))
print(len(Pobservation))

# initialization
photsys = 'gaiaDR3'
model = 'parsec'
imf = 'salpeter55'
imf_inst = IMF(imf)
parsec_inst = Parsec(photsyn=photsys)
isoc_inst = Isoc(parsec_inst)
binmethod = BinMRD()
photerr = Individual(model, photsys, Pobservation)
synstars_inst = SynStars(model, photsys, imf_inst, binmethod, photerr)

step = (0.05, 0.05)  # logage, mh
logage_step, mh_step = step

n_stars = 1060
theta = 8.0, 0.0, 5.5, 0.35, 0.2, 2.3

# samples = synstars_inst(theta, n_stars, isoc_inst)
# samples = samples.rename(columns={'G': 'Gmag', 'BP': 'BPmag', 'RP': 'RPmag'})
# samples = samples[(samples['Gmag'] < max(Pobservation['Gmag'])) &
#                   (samples['Gmag'] > min(Pobservation['Gmag'])) &
#                   ((samples['BPmag'] - samples['RPmag']) < max(Pobservation['BPmag'] - Pobservation['RPmag'])) &
#                   ((samples['BPmag'] - samples['RPmag']) > min(Pobservation['BPmag'] - Pobservation['RPmag']))]
# joblib.dump(samples,
#             '/Users/sara/PycharmProjects/starcat/script/grid_likelihood/cl_sample/logage8_mh0_dm5.5_av0.35_fb0.2_alpha0.3.jobib')

n_syn = 50000
times = 2
samples = joblib.load(
    '/Users/sara/PycharmProjects/starcat/script/grid_likelihood/cl_sample/logage8_mh0_dm5.5_av0.35_fb0.2_alpha0.3.jobib')
test_sample = samples

print(len(test_sample))

h2p_inst = Hist2Point4CMD(model, photsys, bin_method='fixed', sample_obs=test_sample)
gaussian_inst = Gaussian(model, photsys, bin_method='fixed', sample_obs=test_sample)
h2h_inst = Hist2Hist4CMD(model, photsys, bin_method='fixed', sample_obs=test_sample)

likelihood_inst = h2h_inst

c = test_sample['BPmag'] - test_sample['RPmag']
m = test_sample['Gmag']
plt.figure(figsize=(4.5, 5))
plt.scatter(c, m, s=3)
plt.ylim((np.max(m) + 0.5, np.min(m) - 0.5))
plt.xlabel('BP - RP')
plt.ylabel('G')
plt.show()
plt.close()


def calculate_lnlike(args):
    t1, t2, t3, t4, t5, t6 = args
    theta_args = t1, t2, t3, t4, t5, t6
    return lnlike(theta_args, step, isoc_inst, likelihood_inst, synstars_inst, n_syn,
                  'MW', times)


if __name__ == "__main__":
    # set grid
    logage_grid = np.arange(6.8, 9.0, 0.5)
    mh_grid = np.arange(-0.5, 0.4, 0.1)
    dm_grid = np.arange(4.0, 8.0, 0.5)
    Av_grid = np.arange(0.0, 2.0, 0.25)
    fb_grid = np.arange(0.1, 0.7, 0.1)
    alpha_grid = np.arange(0.5, 3.5, 0.25)

    # logage_grid = np.array([7.85, 7.90, 7.95, 8.0, 8.05, 8.1, 8.15])
    # mh_grid = np.array([-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15])
    # dm_grid = np.array([5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8])
    # Av_grid = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6])
    # fb_grid = np.array([0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
    # alpha_grid = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6])

    logage_len = len(logage_grid)
    mh_len = len(mh_grid)
    dm_len = len(dm_grid)
    Av_len = len(Av_grid)
    fb_len = len(fb_grid)
    alpha_len = len(alpha_grid)
    total_num = logage_len * mh_len * dm_len * Av_len * fb_len * alpha_len

    print(f'logage len: {logage_len}, grid: {logage_grid}')
    print(f'mh len: {mh_len}, grid: {mh_grid}')
    print(f'dm len: {dm_len}, grid: {dm_grid}')
    print(f'Av len: {Av_len}, grid: {Av_grid}')
    print(f'fb len: {fb_len}, grid: {fb_grid}')
    print(f'alpha len: {alpha_len}, grid: {alpha_grid}')

    print(f'total samples: '
          f'{logage_len}x{mh_len}x{dm_len}x{Av_len}x{fb_len}x{alpha_len}'
          f'= {total_num}')
    print(f'total likelihood: {total_num}x{times}={total_num * times}')

    aa, bb, cc, dd, ee, ff = np.indices((logage_len, mh_len, dm_len, Av_len, fb_len, alpha_len))
    aa = aa.ravel()
    bb = bb.ravel()
    cc = cc.ravel()
    dd = dd.ravel()
    ee = ee.ravel()
    ff = ff.ravel()

    logage_vals = logage_grid[aa]
    mh_vals = mh_grid[bb]
    dm_vals = dm_grid[cc]
    Av_vals = Av_grid[dd]
    fb_vals = fb_grid[ee]
    alpha_vals = alpha_grid[ff]

    args_list = []
    for i in range(len(logage_vals)):
        logage_val = logage_vals[i]
        mh_val = mh_vals[i]
        dm_val = dm_vals[i]
        Av_val = Av_vals[i]
        fb_val = fb_vals[i]
        alpha_val = alpha_vals[i]
        args_list.append((logage_val, mh_val, dm_val, Av_val, fb_val, alpha_val))

    # check args_list
    args_list_consistent = True
    for i in range(len(args_list)):
        args_list_i = args_list[i]
        # 检查参数组合和索引数组中对应位置的元素是否一致
        if args_list_i != (logage_grid[aa[i]], mh_grid[bb[i]], dm_grid[cc[i]],
                           Av_grid[dd[i]], fb_grid[ee[i]], alpha_grid[ff[i]]):
            args_list_consistent = False
            break
    if args_list_consistent:
        print("参数顺序与索引数组顺序一致")
    else:
        print("参数顺序与索引数组顺序不一致", i)

    joint_lnlike = np.zeros((logage_len, mh_len, dm_len, Av_len, fb_len, alpha_len))

    with Pool(cpu_count()) as p:
        with tqdm(total=len(args_list)) as progress_bar:
            results = []
            for result in p.imap(calculate_lnlike, args_list):
                results.append(result)
                progress_bar.update(1)

    results = np.array(results)
    joint_lnlike[aa, bb, cc, dd, ee, ff] = results
    print(results)
    print(np.min(joint_lnlike), np.max(joint_lnlike))
    joint_lnlike_noinf = joint_lnlike[joint_lnlike != -np.inf]
    print(f'[{len(joint_lnlike_noinf)}/{total_num}]', np.min(joint_lnlike_noinf), np.max(joint_lnlike_noinf))
