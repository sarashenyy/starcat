import joblib
import numpy as np
from joblib import Parallel, delayed
from joblib_progress import joblib_progress

# from matplotlib import gridspec
# ? when running on python console
# from script.corner_tests.draw_corner import draw_corner
# ? when running on terminal
from draw_corner import draw_corner
from starcat import (config,
                     Isoc, Parsec, IMF, BinMRD, CSSTsim,
                     SynStars,
                     lnlike_5p, Hist2Hist4CMD, Hist2Point4CMD)

dir = '/home/shenyueyue/Projects/starcat/data/corner/demo_age9/'
sample_val_path = dir + 'sample_val.joblib'
# samplefig_path = dir + 'sample_val_CMD.png'
h2h_cmd_path = dir + 'h2h_cmd(6).joblib'
h2p_cmd_path = dir + 'h2p_cmd(6).joblib'
h2h_cmd_fig = dir + 'h2h_cmd(6).png'
h2p_cmd_fig = dir + 'h2p_cmd(6).png'

# !define instance from starcat
n_stars = 5000
bins = 50
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
synstars_inst = SynStars(model, photsys, imf_inst, binmethod, photerr)
# ?init LikelihoodFunc
h2h_cmd_inst = Hist2Hist4CMD(model, photsys, bins)
h2p_cmd_inst = Hist2Point4CMD(model, photsys, bins)

# !create synthetic cluster for validation
logage_val = 9.
mh_val = 0.
dist_val = 780.
Av_val = 0.5
fb_val = 0.5
n_val = 1500
theta_val = logage_val, mh_val, dist_val, Av_val, fb_val

source = config.config[model][photsys]
bands = source['bands']
mag = source['mag']  # list
color = source['color']
mag_max = source['mag_max']  # list

# ?synthetic isochrone (distance and Av added)
isoc_ori = isoc_inst.get_isoc(photsys, logage=logage_val, mh=mh_val)
isoc_val = synstars_inst.get_observe_isoc(isoc_ori, dist_val, Av_val)

# ?synthetic cluster sample (without error added)
sample_val_noerr = synstars_inst.sample_stars(isoc_val, n_val, fb_val)

# ?synthetic cluster sample (with phot error)
sample_val = synstars_inst(theta_val, n_val, isoc_ori)

# * save sample data
joblib.dump(sample_val, sample_val_path)

#####################################################
# # ! draw sample_val schematic
# phase = source['phase']
# color_list = ['grey', 'green', 'orange', 'red', 'blue', 'skyblue', 'pink', 'purple', 'grey', 'black']
#
# fig = plt.figure(figsize=(8, 4))
# gs = gridspec.GridSpec(1, 3)
#
# # ? isoc_val
# c = isoc_val[color[0]] - isoc_val[color[1]]
# m = isoc_val[mag]
#
# ax_isoc = plt.subplot(gs[0, 0])
# for j, element in enumerate(phase):
#     index = isoc_val['phase'] == element
#     ax_isoc.plot(c[index], m[index], color=color_list[j])
# ax_isoc.axhline(mag_max, color='r', linestyle=':', label=f'{mag_max}(mag)')
# ax_isoc.invert_yaxis()
# ax_isoc.grid(True, linestyle='--')
# ax_isoc.set_xlabel('g - i')
# ax_isoc.set_ylabel('i')
# ax_isoc.legend(frameon=False)
#
# # ? sample_val_noerr
# single_val_noe = sample_val_noerr[sample_val_noerr['q'].isna()]
# binary_val_noe = sample_val_noerr[~sample_val_noerr['q'].isna()]
#
# ax_noe = plt.subplot(gs[0, 1])
# ax_noe.scatter(single_val_noe[color[0]] - single_val_noe[color[1]], single_val_noe[mag],
#                s=0.5, alpha=0.5, label='single')
# ax_noe.scatter(binary_val_noe[color[0]] - binary_val_noe[color[1]], binary_val_noe[mag],
#                s=0.5, alpha=0.5, label='binary')
# # ax_noe.plot(isoc_val[color[0]]-isoc_val[color[1]], isoc_val[mag],
# #             c='k', linewidth=0.5, linestyle='--')
# ax_noe.invert_yaxis()
# ax_noe.set_ylim(25.5, min(sample_val_noerr[mag]) - 0.1)
# ax_noe.set_xlim(-1.1, 1)
# ax_noe.set_xlabel('g - i')
#
# # ? sample_val
# single_val = sample_val[sample_val['q'].isna()]
# binary_val = sample_val[~sample_val['q'].isna()]
#
# ax_val = plt.subplot(gs[0, 2])
# ax_val.scatter(single_val[color[0]] - single_val[color[1]], single_val[mag],
#                s=0.5, alpha=0.5, label='single')
# ax_val.scatter(binary_val[color[0]] - binary_val[color[1]], binary_val[mag],
#                s=0.5, alpha=0.5, label='binary')
# ax_val.plot(isoc_val[color[0]] - isoc_val[color[1]], isoc_val[mag],
#             c='k', linewidth=0.5, linestyle='--')
# ax_val.invert_yaxis()
# ax_val.set_ylim(25.5, min(sample_val[mag]) - 0.1)
# ax_val.set_xlim(-1.1, 1)
# ax_val.set_xlabel('g - i')
# ax_val.legend(frameon=False)
#
# # * save sample fig
# plt.savefig(samplefig_path, bbox_inches='tight')
#####################################################

# start_time = time.time()

# ! define theta range
logage_step = 0.1  # 0.1
mh_step = 0.1  # 0.05
step = (logage_step, mh_step)

logage = np.arange(8.5, 9.5, logage_step)
mh = np.arange(-0.9, 0.7, mh_step)
dist = np.arange(750, 850, 10)  # 10
Av = np.arange(0., 1., 0.1)  # 0.1
fb = np.arange(0.2, 1., 0.1)  # 0.1
times = 2
# num_repeats = np.arange(times)  # 第0～times-1次
# print(len(logage), len(mh), len(dist), len(Av), len(fb))


# ! calculate joint distribution
ii, jj, aa, bb, cc, tt = np.indices((len(logage), len(mh), len(dist), len(Av), len(fb), times))
ii = ii.ravel()
jj = jj.ravel()
aa = aa.ravel()
bb = bb.ravel()
cc = cc.ravel()
tt = tt.ravel()

logage_vals = logage[ii]
mh_vals = mh[jj]
dist_vals = dist[aa]
Av_vals = Av[bb]
fb_vals = fb[cc]
# num_repeats_vals = num_repeats[tt]

# joint_lnlike_h2h_cmd = np.zeros((len(logage), len(mh), len(dist), len(Av), len(fb)))
# joint_lnlike_h2p_cmd = np.zeros((len(logage), len(mh), len(dist), len(Av), len(fb)))
repeat_results_h2h_cmd = np.zeros((len(logage), len(mh), len(dist), len(Av), len(fb), times))
repeat_results_h2p_cmd = np.zeros((len(logage), len(mh), len(dist), len(Av), len(fb), times))


def h2h_cmd_wrapper(theta_5p):
    h2h_cmd = lnlike_5p(theta_5p, step, isoc_inst, h2h_cmd_inst, synstars_inst, n_stars, sample_val)
    return h2h_cmd


def h2p_cmd_wrapper(theta_5p):
    h2p_cmd = lnlike_5p(theta_5p, step, isoc_inst, h2p_cmd_inst, synstars_inst, n_stars, sample_val)
    return h2p_cmd


def compute_h2h_h2p(l, m, d, A, f):
    h2h_cmd = lnlike_5p((l, m, d, A, f),
                        step, isoc_inst, h2h_cmd_inst, synstars_inst, n_stars, sample_val)
    h2p_cmd = lnlike_5p((l, m, d, A, f),
                        step, isoc_inst, h2p_cmd_inst, synstars_inst, n_stars, sample_val)
    return h2h_cmd, h2p_cmd


# test before parallel
# l, m, d, A, f = logage_vals[22024], mh_vals[22024], dist_vals[22024], Av_vals[22024], fb_vals[22024]
# print(l, m, d, A, f)
# t1 = lnlike_5p((l, m, d, A, f),
#                     step, isoc_inst, h2h_cmd_inst, synstars_inst, n_stars, sample_val)
# t2 = lnlike_5p((l, m, d, A, f),
#                     step, isoc_inst, h2p_cmd_inst, synstars_inst, n_stars, sample_val)
# print(t1, t2)

# 并行计算联合分布
n_jobs = -1
try:
    with joblib_progress("Calculating lnlikes...", total=len(ii)):
        results_h2h_cmd, results_h2p_cmd = zip(*Parallel(n_jobs=n_jobs)(
            delayed(compute_h2h_h2p)(l, m, d, A, f) for l, m, d, A, f in
            zip(logage_vals, mh_vals, dist_vals, Av_vals, fb_vals)
        ))
# try:
#     with joblib_progress("Calculating lnlikes...", total=len(ii)):
#         results_h2h_cmd, results_h2p_cmd = zip(*Parallel(n_jobs=n_jobs)(
#             delayed(compute_h2h_h2p)(l, m) for l, m in
#             zip(logage_vals, mh_vals)
#         ))

except:
    repeat_results_h2h_cmd[ii, jj, aa, bb, cc, tt] = results_h2h_cmd
    repeat_results_h2p_cmd[ii, jj, aa, bb, cc, tt] = results_h2p_cmd
    joblib.dump(repeat_results_h2h_cmd, h2h_cmd_path)
    joblib.dump(repeat_results_h2p_cmd, h2p_cmd_path)

    text_content = (f'parallel stopped on the {len(results_h2h_cmd)} lnlike for Hist2HistCMD\n'
                    f'parallel stopped on the {len(results_h2p_cmd)} lnlike for Hist2PointCMD\n')
    with open(dir + 'output.txt', 'w') as file:
        file.write(text_content)
    print("details saved to output.txt")

else:
    # joint_lnlike_h2h_cmd[ii, jj, aa, bb, cc] = results_h2h_cmd
    # joint_lnlike_h2p_cmd[ii, jj, aa, bb, cc] = results_h2p_cmd
    repeat_results_h2h_cmd[ii, jj, aa, bb, cc, tt] = results_h2h_cmd
    repeat_results_h2p_cmd[ii, jj, aa, bb, cc, tt] = results_h2p_cmd
    joint_lnlike_h2h_cmd = np.mean(repeat_results_h2h_cmd, axis=5)
    joint_lnlike_h2p_cmd = np.mean(repeat_results_h2p_cmd, axis=5)

    # 120000 costs 9min
    # 120000(lnlike one)*2(h2h,h2p) costs 9min
    # 120000(50 times) costs 8847s=147min=2.45h
    # end_time = time.time()
    # run_time = end_time - start_time
    # print(f"运行时间: {run_time} 秒")

    # * save lnlike data
    joblib.dump(joint_lnlike_h2h_cmd, h2h_cmd_path)
    joblib.dump(joint_lnlike_h2p_cmd, h2p_cmd_path)
    joblib.dump(repeat_results_h2h_cmd,
                dir + 'repeat_h2h.joblib')
    joblib.dump(repeat_results_h2p_cmd,
                dir + 'repeat_h2p.joblib')
    # joint_lnlike_h2h_cmd = joblib.load(h2h_cmd_path)
    # joint_lnlike_h2p_cmd = joblib.load(h2p_cmd_path)
    # repeat_results_h2h_cmd = joblib.load('/home/shenyueyue/Projects/starcat/data/corner/progress_test/repeat_h2h.joblib')
    # repeat_results_h2p_cmd = joblib.load('/home/shenyueyue/Projects/starcat/data/corner/progress_test/repeat_h2p.joblib')

    # ! draw corner plot
    truth = list(theta_val)
    parameters = [logage, mh, dist, Av, fb]

    label = ['$log_{10}{\\tau}$', '[M/H]', '$d $(kpc)', '$A_{v}$', '$f_{b}$']
    info_h2h_cmd = [photsys, 'kroupa01', 'BinMRD(uniform distribution)', 'Hist2Hist4CMD(6)', n_stars]
    info_h2p_cmd = [photsys, 'kroupa01', 'BinMRD(uniform distribution)', 'Hist2Point4CMD(6)', n_stars]

    draw_corner(truth=truth, parameters=parameters, ln_joint_distribution=joint_lnlike_h2h_cmd,
                label=label, info=info_h2h_cmd, savefig_path=h2h_cmd_fig)
    draw_corner(truth=truth, parameters=parameters, ln_joint_distribution=joint_lnlike_h2p_cmd,
                label=label, info=info_h2p_cmd, savefig_path=h2p_cmd_fig)
