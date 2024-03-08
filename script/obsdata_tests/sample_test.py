import time

import dynesty
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from starcat import (Isoc, Parsec, IMF,
                     BinMRD,
                     Individual, SynStars,
                     lnlike, Hist2Point4CMD)

# re-defining plotting defaults

plt.style.use('/Users/sara/PycharmProjects/starcat/data/mystyle.mplstyle')


# plt.style.use('/home/shenyueyue/Projects/starcat/data/mystyle.mplstyle')


def loglikelihood(x, step, isoc_inst, likelihood_inst, synstars_inst, n_syn, observation):
    return lnlike(x, step, isoc_inst, likelihood_inst, synstars_inst,
                  n_syn, observation, 'MW', 10)


def prior_transform(u):
    """Transforms the uniform random variables `u ~ Unif[0., 1.)`
    to the parameters of interest."""
    x = u.copy()  # copy u
    x_range = [[6.7, 9.0], [-2.0, 0.4],
               [3.0, 10.], [0.0, 2.0],
               [0.0, 1.0], [1.0, 2.8]]
    for i, xrange in enumerate(x_range):
        x[i] = (1 - u[i]) * xrange[0] + u[i] * xrange[1]
    return x


# def prior_gaussian(u):
#     """Transforms the uniform random variables `u ~ Unif[0., 1.)`
#     to the parameters of interest.
#
#     All parameters have Gaussian Prior.
#     """
#     x = u.copy()  # copy u
#
#     mean = [8.4, 0.0, 18.5, 0.4, 0.5, 2.35]
#     std = [0.5, 0.1, 0.8, 0.5, 0.1, 0.2]
#     x_range = [[6.7, 10.0], [-2.0, 0.4],
#                [15, 22], [0.0, 2.0],
#                [0.2, 1.0], [1.6, 3.0]]
#
#     for i, (m, s, r) in enumerate(zip(mean, std, x_range)):
#         low, high = r[0], r[1]  # lower and upper bounds
#         low_n, high_n = (low - m) / s, (high - m) / s  # standardize
#         x[i] = scipy.stats.truncnorm.ppf(u[i], low_n, high_n, loc=m, scale=s)
#
#     return x


# def prior_M(u):
#     """Transforms the uniform random variables `u ~ Unif[0., 1.)`
#     to the parameters of interest.
#
#     Only [M/H] has Gaussian prior.
#     """
#     x = u.copy()  # copy u
#     # uniform
#     x_range = [[6.7, 10.0], [-2.0, 0.4],
#                [15, 22], [0.0, 2.0],
#                [0.2, 1.0], [1.6, 3.0]]
#     for i, xrange in enumerate(x_range):
#         if i != 1:
#             x[i] = (1 - u[i]) * xrange[0] + u[i] * xrange[1]
#     # Gaussian [M/H]
#     m, s = 0.0, 0.1
#     low, high = -2.0, 0.4  # lower and upper bounds
#     low_n, high_n = (low - m) / s, (high - m) / s  # standardize
#     x[1] = scipy.stats.truncnorm.ppf(u[i], low_n, high_n, loc=m, scale=s)
#     return x


# for photerr

# read data pandas
file_path = '/home/shenyueyue/Projects/starcat/data/FuXiaoting/In100Myr/Huluwa_1 NewMass.csv'
obs_sample = pd.read_csv(file_path,
                         usecols=['e_Gmag', 'e_BPmag', 'e_RPmag', 'phot_g_mean_mag', 'phot_bp_mean_mag',
                                  'phot_rp_mean_mag'])

Pobservation = obs_sample.rename(
    columns={'phot_g_mean_mag': 'Gmag', 'phot_bp_mean_mag': 'BPmag', 'phot_rp_mean_mag': 'RPmag'})

# read data npy
# file_path = '/home/shenyueyue/Projects/starcat/data/Almeida23/membership_edr3/Melotte_22_data_stars.npy'
# file_path = '/Users/sara/PycharmProjects/starcat/data/Almeida23/membership_edr3/Melotte_22.npy'
file_path = '/Users/sara/PycharmProjects/starcat/data/Almeida23/membership_edr3/Melotte_22_data_stars.npy'
obs_sample = np.load(file_path)
selected_columns = ['Gmag', 'BPmag', 'RPmag', 'e_Gmag', 'e_BPmag', 'e_RPmag']
Pobservation = pd.DataFrame(obs_sample[selected_columns])

Pobservation = Pobservation[(Pobservation['BPmag'] - Pobservation['RPmag']) > -0.2]
Pobservation = Pobservation[Pobservation['Gmag'] < 18.]
c = Pobservation['BPmag'] - Pobservation['RPmag']
sigma_c = np.sqrt((Pobservation['e_BPmag']) ** 2 + (Pobservation['e_RPmag']) ** 2)
Pobservation = Pobservation[sigma_c < np.std(sigma_c) * 3]

print(max(Pobservation['Gmag']), max(Pobservation['BPmag']), max(Pobservation['RPmag']))
print(len(Pobservation))

fig, ax = plt.subplots(figsize=(4, 5))
c = Pobservation['BPmag'] - Pobservation['RPmag']
m = Pobservation['Gmag']
ax.scatter(c, m, s=3, c='blue', label='obs')
# ax.scatter(samples['BP']-samples['RP'], samples['G'], s=3, c='green', label='ref')
# ax.scatter(samples['BP']-samples['RP'], samples['G'], s=3, c='orange', label='best fit')
ax.legend()
ax.set_ylim(max(m) + 1, min(m) - 1)
ax.set_ylabel('G')
ax.set_xlabel('BP - RP')
fig.show()

# initialization
photsys = 'gaiaDR3'
model = 'parsec'
# imf = 'salpeter55'
imf = 'kroupa01'
imf_inst = IMF(imf)
parsec_inst = Parsec(photsyn=photsys)  # 初始化带有photsyn参数，预先加载等龄线库至内存
binmethod = BinMRD()
photerr = Individual(model, photsys, Pobservation)
isoc_inst = Isoc(parsec_inst)
synstars_inst = SynStars(model, photsys,
                         imf_inst, binmethod, photerr)

# logage, mh, dm, Av, fb, alpha = 8.00, 0.0, 5.65, 0.349, 0.18, 2.61 melotte_22 previous research
# logage, mh, dm, Av, fb, alpha = 7.25, 0.0, 7.38, 0.35, 0.3, 2.3 #huluwa_1
logage, mh, dm, Av, fb, alpha = 7.20, 0.1, 7.69, 0.26, 0.37, 2.77  # huluwa_1 this work
# logage, mh, dm, Av, fb, alpha = 8.08, 0.33, 6.0, 0.13, 0.39, 2.19   #melotte_22 this work
theta = logage, mh, dm, Av, fb, alpha
step = (0.05, 0.05)  # logage, mh
logage_step, mh_step = step

n_stars = len(Pobservation)
n_syn = 50000
# bins = 50
h2p_cmd_inst = Hist2Point4CMD(model, photsys, bin_method='knuth')
# sahaw_inst = SahaW(model, photsys, bin_method='knuth')
# energy_inst = EnergyDistance(model, photsys, bin_method='knuth')
# kde_inst = GaussianKDE(model, photsys)

likelihood_inst = h2p_cmd_inst

logage, mh, dm, Av, fb, alpha = 7.25, 0.0, 7.38, 0.35, 0.3, 2.3
theta = logage, mh, dm, Av, fb, alpha
samples = synstars_inst(theta, n_stars, isoc_inst)
# Pobservation = samples
# Pobservation = samples.rename(columns={'G': 'Gmag', 'BP': 'BPmag', 'RP': 'RPmag'})
# observation = samples  # samples[(samples['g'] < 26.0)]  # samples[(samples['g'] < 25.5) & (samples['i'] < 25.5)]
# observation = samples.rename(columns={'G': 'Gmag', 'BP': 'BPmag', 'RP': 'RPmag'})
#
# obs_isoc = isoc_inst.get_obsisoc(photsys, dm=dm, Av=Av, logage=logage, mh=mh, logage_step=logage_step,
#                                  mh_step=mh_step)
#
# fig, ax = plt.subplots(figsize=(4, 5))
# bin = observation['mass_sec'].notna()
# c = observation['BPmag'] - observation['RPmag']
# m = observation['Gmag']
#
# ax.scatter(c[bin], m[bin], color='#8E8BFE', s=5, alpha=0.6, label='binary')
# ax.scatter(c[~bin], m[~bin], color='#E88482', s=5, alpha=0.6, label='single')
# ax.plot(obs_isoc['BP'] - obs_isoc['RP'], obs_isoc['G'], color='r')
# ax.set_ylim(max(m) + 0.2, min(m) - 0.5)
# ax.set_xlim(min(c) - 0.2, max(c) + 0.2)
# ax.legend()
# ax.set_title(f'logage={logage}, [M/H]={mh}, DM={dm}, \n'
#              f'Av={Av}, fb={fb}, alpha={alpha}', fontsize=12)
# ax.set_xlabel('BP - RP')
# ax.set_ylabel('G')
#
# fig.show()

samples = synstars_inst(theta, n_syn, isoc_inst)
h2p = lnlike(theta, step, isoc_inst, h2p_cmd_inst, synstars_inst, n_syn, Pobservation, 'MW', 10)
print(h2p)

# fig, ax = plt.subplots(figsize=(4, 4))
# bin = samples['mass_sec'].notna()
# c = samples['BP'] - samples['RP']
# m = samples['G']
# c_obs = observation['BPmag'] - observation['RPmag']
# m_obs = observation['Gmag']
#
# ax.scatter(c[bin], m[bin], color='#8E8BFE', s=5, alpha=0.6)
# ax.scatter(c[~bin], m[~bin], color='#E88482', s=5, alpha=0.6)
# ax.scatter(c_obs, m_obs, s=1, color='grey')
# ax.invert_yaxis()
# ax.set_xlim(min(c_obs) - 0.2, max(c_obs) + 0.2)
# ax.set_ylim(max(m_obs) + 0.5, min(m_obs) - 0.5)
# ax.set_title(f'logage={logage}, [M/H]={mh}, DM={dm}, \n'
#              f'Av={Av}, fb={fb}, alpha={alpha}', fontsize=12)
# ax.set_xlabel('BP - RP')
# ax.set_ylabel('G')
# ax.text(0.6, 0.5, f'H2P={h2p:.2f}', transform=ax.transAxes, fontsize=11)
#
# fig.show()

# Define the dimensionality of our problem.
ndim = 6
# sampler = dynesty.NestedSampler(loglike, ptform, ndim)
# sampler.run_nested()
# loglike_args = (step, isoc_inst, likelihood_inst, synstars_inst, n_syn, observation, 'LG', 5)
# Set up the dynamic nested sampling run with multiprocessing
start_time = time.time()
with dynesty.pool.Pool(50, loglikelihood, prior_transform,
                       logl_args=(step, isoc_inst, likelihood_inst, synstars_inst, n_syn, Pobservation)) as pool:
    # Statistic
    # sampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform, ndim, pool=pool,
    #                                 nlive=1000, bound='multi', sample='rwalk')
    # sampler.run_nested(dlogz=0.1, checkpoint_file='H2P.6P.improve.save')
    # dlogz=0.01  THE END condition, checkpoint_file='dynesty.save'

    # Dynamic
    dsampler = dynesty.DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim, pool=pool,
                                            bound='multi', sample='rwalk')
    # sample='rslice'， first_update={'min_eff':25.}
    # dsampler.run_nested(dlogz_init=20, maxiter=10000,
    #                     nlive_init=1000,
    #                     wt_kwargs={'pfrac': 1.0},
    #                     checkpoint_file='Melotte_22_H2Pv1.dsave')  # maxiter_batch=2000, nlive_init=500
    dsampler.run_nested(dlogz_init=10, nlive_init=1000, maxiter=30000, maxiter_batch=8000,
                        wt_kwargs={'pfrac': 1.0},
                        checkpoint_file='Huluwa_1_OBS_H2Pv1.dsave')  # maxiter_batch=2000, nlive_init=500
    # n_effective=10000,
    # dlogz_init=0.05, nlive_init=500, n_effective=20000, wt_kwargs={'pfrac': 1.0}, use_stop=False
    # nlive_init=500, nlive_batch=100, maxiter_init=10000, maxiter_batch=1000, maxbatch=10,  n_effective=5000,
end_time = time.time()
runtime = end_time - start_time
print(f'time: {runtime / 3600:.2f}h')
filename = 'Huluwa_1_OBS.dsample'
# dir = '/Users/sara/PycharmProjects/starcat/script/dynesty_tests/'
dir = '/home/shenyueyue/Projects/starcat/script/obsdata_tests/dsample/'
path = dir + filename
joblib.dump(dsampler, path)

# ndim = 6
# start_time = time.time()
# with dynesty.pool.Pool(50, loglikelihood, prior_transform,
#                        logl_args=(step, isoc_inst, likelihood_inst, synstars_inst, n_syn, Pobservation)) as pool:
#     dsampler = dynesty.NestedSampler.restore('/home/shenyueyue/Projects/starcat/Melotte_22_OBS_H2Pv1.dsave', pool=pool)
#     dsampler.run_nested(dlogz_init=1, maxiter=30000, resume=True,
#                         checkpoint_file='Melotte_22_OBS_H2Pv1.dsave')
# end_time = time.time()
# runtime = end_time - start_time
# print(f'time: {runtime / 3600:.2f}h')

# dsampler = dynesty.NestedSampler.restore('/home/shenyueyue/Projects/starcat/Huluwa_1_OBS_H2Pv1.dsave')
# results = dsampler.results
#
# label = ['log(age)', '[M/H]', 'DM', '$A_v$', '$f_b$', '$\\alpha$']
# span = ([6.7, 10], [-2.0, 0.4], [17.5, 22], [0.0, 2.0], [0.2, 1.0], [1.6, 3.0])
# # summary (run) plot
# fig, axes = dyplot.runplot(results)
# fig.tight_layout()
# fig.show()
#
# # Trace Plots: generate a trace plot showing the evolution of particles
# # (and their marginal posterior distributions) in 1-D projections.
# # colored by importance weight
# # Highlight specific particle paths (shown above) to inspect the behavior of individual particles.
# # (These can be useful to qualitatively identify problematic behavior such as strongly correlated samples.)
# fig, axes = dyplot.traceplot(results, truths=theta,
#                              labels=label,
#                              quantiles=(0.16, 0.5, 0.85),
#                              title_quantiles=(0.16, 0.5, 0.85),
#                              truth_color='black', show_titles=True,
#                              title_kwargs={'fontsize': 24},
#                              trace_cmap='viridis', connect=True,
#                              connect_highlight=range(5)
#                              )  # fig=plt.subplots(6, 2, figsize=(15, 20))
# fig.tight_layout()
# # fig.subplots_adjust(wspace=0.1, hspace=0.8)
# fig.show()
#
# # Corner Points
# # kde=True: colored according to their estimated posterior mass
# # kde=False: colored according to raw importance weights
# fg, ax = dyplot.cornerpoints(results, cmap='plasma', truths=theta,
#                              truth_color='red',
#                              labels=label,
#                              kde=True)
# fg.show()
#
# # Corner Plot
# # DEFAULT quantiles=(0.025, 0.5, 0.975)
# fg, ax = dyplot.cornerplot(results, color='blue', truths=theta,
#                            quantiles=(0.16, 0.5, 0.85),
#                            title_quantiles=(0.16, 0.5, 0.85),
#                            labels=label,
#                            truth_color='black', show_titles=True,
#                            max_n_ticks=3)  # quantiles=None
# fg.show()
#
# # initialize figure
# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# # plot 6 snapshots over the course of the run
# for i, a in enumerate(axes.flatten()):
#     it = int((i + 1) * results.niter / 8.)
#     # overplot the result onto each subplot
#     temp = dyplot.boundplot(results, it=it,
#                             prior_transform=prior_transform,
#                             max_n_ticks=3, show_live=True,
#                             dims=(1, 5), labels=['[M/H]', '$\\alpha$'],
#                             # span=[(6.7, 10), (-2.0, 0.4)],
#                             fig=(fig, a))
#     a.set_title('Iteration {0}'.format(it), fontsize=26)
# fig.tight_layout()
# fig.show()
#
# fig, axes = dyplot.cornerbound(results, it=15000,
#                                prior_transform=prior_transform,
#                                show_live=True)
# fig.show()
