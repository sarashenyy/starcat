import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib import gridspec

from starcat import (config,
                     Isoc, Parsec, IMF, BinMRD, CSSTsim,
                     SynStars,
                     lnlike_5p, Hist2Hist4CMD, Hist2Point4CMD, Hist2Hist4Bands)

# !define instance from starcat
n_stars = 5000
bins = 50
step_lnlike = 0.5
photsys = 'CSST'
model = 'parsec'

# ?init isochrone
parsec_inst = Parsec()
isoc_inst = Isoc(parsec_inst)
# ?init IMF
imf_inst = IMF('kroupa01')
# ?init Binmethod & photometric error system
binmethod = BinMRD()
photerr = CSSTsim(model)
# ?init SynStars
synstars_inst = SynStars(model, photsys, imf_inst, n_stars, binmethod, photerr)
# ?init LikelihoodFunc
h2h_cmd_inst = Hist2Hist4CMD(model, photsys, bins)
h2p_cmd_inst = Hist2Point4CMD(model, photsys, bins)
# h2h_bds_inst = Hist2Hist4Bands(model, photsys, step = step_lnlike)
h2h_bds_inst = Hist2Hist4Bands(model, photsys)

# !create synthetic cluster for validation
logage_val = 7.
mh_val = 0.
dist_val = 780.
Av_val = 0.5
fb_val = 0.5
n_val = 1300
theta_val = logage_val, mh_val, dist_val, Av_val, fb_val

source = config.config[model][photsys]
bands = source['bands']
mag = source['mag']
color = source['color']
mag_max = source['mag_max']

# ?synthetic isochrone (distance and Av added)
synstars_val = SynStars(model, photsys, imf_inst, n_val, binmethod, photerr)
isoc_ori = isoc_inst.get_isoc(photsys, logage=logage_val, mh=mh_val)
isoc_val = synstars_val.get_observe_isoc(isoc_ori, dist_val, Av_val)

# ?synthetic cluster sample (without error added)
sample_val_noerr = synstars_val.sample_stars(isoc_val, fb_val)

# ?synthetic cluster sample (with phot error)
sample_val = synstars_val(theta_val, isoc_ori)
sample_val = sample_val[sample_val[mag] <= 25.5]

# %%
# ! draw sample_val schematic
phase = source['phase']
color_list = ['grey', 'green', 'orange', 'red', 'blue', 'skyblue', 'pink', 'purple', 'grey', 'black']

fig = plt.figure(figsize=(8, 4))
gs = gridspec.GridSpec(1, 3)

# ? isoc_val
c = isoc_val[color[0]] - isoc_val[color[1]]
m = isoc_val[mag]

ax_isoc = plt.subplot(gs[0, 0])
for j, element in enumerate(phase):
    index = isoc_val['phase'] == element
    ax_isoc.plot(c[index], m[index], color=color_list[j])
ax_isoc.axhline(mag_max, color='r', linestyle=':', label=f'{mag_max}(mag)')
ax_isoc.invert_yaxis()
ax_isoc.grid(True, linestyle='--')
ax_isoc.set_xlabel('g - i')
ax_isoc.set_ylabel('i')
ax_isoc.legend(frameon=False)

# ? sample_val_noerr
single_val_noe = sample_val_noerr[sample_val_noerr['q'].isna()]
binary_val_noe = sample_val_noerr[~sample_val_noerr['q'].isna()]

ax_noe = plt.subplot(gs[0, 1])
ax_noe.scatter(single_val_noe[color[0]] - single_val_noe[color[1]], single_val_noe[mag],
               s=0.5, alpha=0.5, label='single')
ax_noe.scatter(binary_val_noe[color[0]] - binary_val_noe[color[1]], binary_val_noe[mag],
               s=0.5, alpha=0.5, label='binary')
# ax_noe.plot(isoc_val[color[0]]-isoc_val[color[1]], isoc_val[mag],
#             c='k', linewidth=0.5, linestyle='--')
ax_noe.invert_yaxis()
ax_noe.set_ylim(25.5, min(sample_val_noerr[mag]) - 0.1)
ax_noe.set_xlim(-1.1, 1)
ax_noe.set_xlabel('g - i')

# ? sample_val
single_val = sample_val[sample_val['q'].isna()]
binary_val = sample_val[~sample_val['q'].isna()]

ax_val = plt.subplot(gs[0, 2])
ax_val.scatter(single_val[color[0]] - single_val[color[1]], single_val[mag],
               s=0.5, alpha=0.5, label='single')
ax_val.scatter(binary_val[color[0]] - binary_val[color[1]], binary_val[mag],
               s=0.5, alpha=0.5, label='binary')
ax_val.plot(isoc_val[color[0]] - isoc_val[color[1]], isoc_val[mag],
            c='k', linewidth=0.5, linestyle='--')
ax_val.invert_yaxis()
ax_val.set_ylim(25.5, min(sample_val[mag]) - 0.1)
ax_val.set_xlim(-1.1, 1)
ax_val.set_xlabel('g - i')
ax_val.legend(frameon=False)

plt.show()

# %%
import time

start_time = time.time()

# ! define theta range
logage_step = 0.1
mh_step = 0.05
step = (logage_step, mh_step)

logage = np.arange(6.6, 8., logage_step)
mh = np.arange(-0.5, 0.5, mh_step)
dist = np.arange(750, 850, 10)
Av = np.arange(0., 1., 0.1)
fb = np.arange(0.2, 1., 0.1)
times = 1

# ! calculate joint distribution
ii, jj, aa, bb, cc = np.indices((len(logage), len(mh), len(dist), len(Av), len(fb)))
ii = ii.ravel()
jj = jj.ravel()
aa = aa.ravel()
bb = bb.ravel()
cc = cc.ravel()

logage_vals = logage[ii]
mh_vals = mh[jj]
dist_vals = dist[aa]
Av_vals = Av[bb]
fb_vals = fb[cc]

joint_lnlike_h2h_cmd = np.zeros((len(logage), len(mh), len(dist), len(Av), len(fb)))
joint_lnlike_h2p_cmd = np.zeros((len(logage), len(mh), len(dist), len(Av), len(fb)))
joint_lnlike_h2h_bds = np.zeros((len(logage), len(mh), len(dist), len(Av), len(fb)))


def h2h_cmd_wrapper(theta_5p):
    h2h_cmd = lnlike_5p(theta_5p, step, isoc_inst, h2h_cmd_inst, synstars_inst, sample_val, times)
    return h2h_cmd


def h2p_cmd_wrapper(theta_5p):
    h2p_cmd = lnlike_5p(theta_5p, step, isoc_inst, h2p_cmd_inst, synstars_inst, sample_val, times)
    return h2p_cmd


def h2h_bds_wrapper(theta_5p):
    h2h = lnlike_5p(theta_5p, step, isoc_inst, h2h_bds_inst, synstars_inst, sample_val, times)
    return h2h


def compute_h2h_h2p(l, m, d, A, f):
    h2h_cmd = lnlike_5p((l, m, d, A, f), step, isoc_inst, h2h_cmd_inst, synstars_inst, sample_val, times)
    h2p_cmd = lnlike_5p((l, m, d, A, f), step, isoc_inst, h2p_cmd_inst, synstars_inst, sample_val, times)
    h2h_bds = lnlike_5p((l, m, d, A, f), step, isoc_inst, h2h_bds_inst, synstars_inst, sample_val, times)
    return h2h_cmd, h2p_cmd, h2h_bds


# def if_pickle_good(l, m):
#     tt = isoc_inst.get_isoc(photsys, logage=l, mh=m, logage_step=logage_step, mh_step=mh_step)
#     return tt
# n_jobs = -1  # 使用所有可用的处理器核心
# results_tt = Parallel(n_jobs=n_jobs)(
#     delayed(if_pickle_good)(l, m) for l, m in itertools.product(logage, mh)
# )

# 并行计算联合分布
n_jobs = -1
# results_h2h_cmd = Parallel(n_jobs=n_jobs)(
#     delayed(h2h_cmd_wrapper)((l, m, d, A, f)) for l, m, d, A, f in
#     zip(logage_vals, mh_vals, dist_vals, Av_vals, fb_vals)
# )
# results_h2p = Parallel(n_jobs=n_jobs)(
#     delayed(h2p_wrapper)((l, m, d, A, f)) for l, m, d, A, f in zip(logage_vals, mh_vals, dist_vals, Av_vals, fb_vals)
# )
# results_h2h_bds = Parallel(n_jobs=n_jobs)(
#     delayed(h2h_bds_wrapper)((l, m, d, A, f)) for l, m, d, A, f in zip(logage_vals, mh_vals, dist_vals, Av_vals, fb_vals)
# )
results_h2h_cmd, results_h2p_cmd, results_h2h_bds = zip(*Parallel(n_jobs=n_jobs)(
    delayed(compute_h2h_h2p)(l, m, d, A, f) for l, m, d, A, f in zip(logage_vals, mh_vals, dist_vals, Av_vals, fb_vals)
))

joint_lnlike_h2h_cmd[ii, jj, aa, bb, cc] = results_h2h_cmd
joint_lnlike_h2p_cmd[ii, jj, aa, bb, cc] = results_h2p_cmd
joint_lnlike_h2h_bds[ii, jj, aa, bb, cc] = results_h2h_bds

# 120000 costs 9min
# 120000(lnlike one)*2(h2h,h2p) costs 9min
# 120000(50 times) costs 8847s=147min=2.45h
end_time = time.time()
run_time = end_time - start_time
print(f"运行时间: {run_time} 秒")


# %%
# ! draw corner plot
truth = list(theta_val)
parameters = [logage, mh, dist, Av, fb]
ln_joint_distribution = joint_lnlike_h2h_cmd
joint_distribution = np.exp(ln_joint_distribution)
num_subplot = len(joint_distribution.shape)
label = ['$log_{10}{\\tau}$', '[M/H]', '$d $(kpc)', '$A_{v}$', '$f_{b}$']
info = [photsys, 'kroupa01', 'BinMRD(uniform distribution)', 'Hist2Hist4Bands', n_stars]


fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(num_subplot, num_subplot)  # (nrows,ncols)
gs.update(wspace=0.06, hspace=0.06)

rows_to_draw = np.arange(num_subplot)
for col in range(num_subplot):
    # ! calculate marginal distribution of single parameter
    # get x axis data
    data_x = parameters[col]
    # get y axis data (marginal distribution of p)
    marginal_p = np.sum(joint_distribution, axis=tuple(set(np.arange(num_subplot)) - {col}))
    # normalize
    marginal_p = marginal_p / np.sum(marginal_p)
    # ? draw diagonal plots with marginal distribution of single parameter
    ax_diagonal = plt.subplot(gs[col, col])
    ax_diagonal.plot(data_x, marginal_p, c='k')
    ax_diagonal.scatter(data_x, marginal_p, c='grey')
    ax_diagonal.axvline(truth[col], c='r', linestyle='--')

    # ax_diagonal.set_xticklabels([])
    # ax_diagonal.set_yticklabels([])
    # ax_diagonal.tick_params(axis='x', direction='in')
    # ax_diagonal.tick_params(axis='y', direction='in')
    #
    # if col == num_subplot - 1:
    #     ax_diagonal.set_xlabel(label[col])
    # if col == 0:
    #     ax_diagonal.set_ylabel(label[col])
    if col != num_subplot - 1:
        # ax_diagonal.get_xaxis().set_visible(False)
        ax_diagonal.set_xticklabels([])
    if col != 0:
        ax_diagonal.get_yaxis().set_visible(False)
        ax_diagonal.set_yticklabels([])
    ax_diagonal.tick_params(axis='x', direction='out')
    ax_diagonal.tick_params(axis='y', direction='out')

    ax_diagonal.set_xlabel(label[col], fontsize=16)
    ax_diagonal.set_ylabel(label[col], fontsize=16)

    # ! calculate marginal distribution of two parameters
    if col < num_subplot - 1:
        rows_to_draw = rows_to_draw[1:]
        for row in rows_to_draw:
            # marginal_pp = np.sum(joint_distribution, axis=tuple(set(np.arange(num_subplot)) - {col, row}))
            # marginal_pp = np.transpose(marginal_pp)
            ln_marginal_pp = np.sum(ln_joint_distribution, axis=tuple(set(np.arange(num_subplot)) - {col, row}))
            ln_marginal_pp = np.transpose(ln_marginal_pp)
            # print(f'row,col={row, col}, axis={tuple(set(np.arange(num_subplot)) - {col, row})}, '
            #       f'marginal_pp={marginal_pp.shape}')
            data_y = parameters[row]
            x_grid, y_grid = np.meshgrid(data_x, data_y)
            # ? draw non diagonal plots with marginal distribution of two parameters
            ax = plt.subplot(gs[row, col])
            ax.scatter(x_grid, y_grid, color='gray', alpha=0.5, s=2)
            # ax.scatter(x_grid, y_grid, c=marginal_pp, cmap='viridis')
            # * NOTE! ax.imshow(pp), pp.shape=(rows_Y, cols_X)
            ax.imshow(ln_marginal_pp, cmap='GnBu', extent=(data_x.min(), data_x.max(), data_y.min(), data_y.max()))
            # * NOTE! ax.contour(X,Y,Z)
            # * NOTE! X and Y must both be 2D with the same shape as Z (e.g. created via numpy.meshgrid)
            ax.contour(x_grid, y_grid, ln_marginal_pp, colors='black', linewidths=0.5, linestyles='-',
                       extent=(data_x.min(), data_x.max(), data_y.min(), data_y.max()))
            ax.set_aspect('auto')
            ax.axvline(truth[col], c='r', linestyle='--')
            ax.axhline(truth[row], c='r', linestyle='--')
            ax.plot(truth[col], truth[row], 'sr')

            # ax.tick_params(axis='x', direction='in')
            # ax.tick_params(axis='y', direction='in')
            # if row == num_subplot - 1:
            #     ax.set_xlabel(label[col])
            # if col == 0:
            #     ax.set_ylabel(label[row])
            if row != num_subplot - 1:
                # ax.get_xaxis().set_visible(False)
                ax.set_xticklabels([])
            if col != 0:
                # ax.get_yaxis().set_visible(False)
                ax.set_yticklabels([])
            ax.tick_params(axis='x', direction='out')
            ax.tick_params(axis='y', direction='out')

            if row == num_subplot - 1:
                ax.set_xlabel(label[col], fontsize=16)
            if col == 0:
                ax.set_ylabel(label[row], fontsize=16)

fig.text(0.5, 0.7, f'Truth: {truth}\n\n'
                   f'Photsys: {info[0]}\nIMF: {info[1]}\nBinary: {info[2]}\nLikelihood: {info[3]}\n'
                   f'Synthetic star number: {info[4]}',
         fontsize=16, ha='left')
plt.show()
