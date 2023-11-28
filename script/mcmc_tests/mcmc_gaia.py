from multiprocessing import Pool

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from starcat import (Isoc, Parsec, IMF,
                     BinMRD, Individual,
                     SynStars,
                     lnlike_5p, Hist2Hist4CMD, Hist2Point4CMD)

# %matplotlib osx

plt.style.use('/Users/sara/PycharmProjects/starcat/data/mystyle.mplstyle')
file_path = '/Users/sara/PycharmProjects/starcat/data/Hunt23/NGC_2682.csv'

# %%
obs_sample = pd.read_csv(file_path, usecols=['Prob', 'Gmag', 'BPmag', 'RPmag', 'G_err', 'BP_err', 'RP_err'])

observation = obs_sample[obs_sample['Prob'] > 0.6]
observation = observation[(observation['BPmag'] - observation['RPmag']) > -0.2]
observation = observation[observation['Gmag'] < 18.]

c = observation['BPmag'] - observation['RPmag']
sigma_c = np.sqrt((observation['BP_err']) ** 2 + (observation['RP_err']) ** 2)
observation = observation[sigma_c < np.std(sigma_c) * 6]

print(max(observation['Gmag']), max(observation['BPmag']), max(observation['RPmag']))
print(len(observation))

fig, ax = plt.subplots(figsize=(3, 4))
ax.scatter(observation['BPmag'] - observation['RPmag'], observation['Gmag'], s=3)
ax.set_ylim(max(observation['Gmag']) + 1, min(observation['Gmag']) - 1)
fig.show()

# %%
# initialization
photsys = 'gaiaDR3'
model = 'parsec'
imf = 'kroupa01'
imf_inst = IMF(imf)
parsec_inst = Parsec()
# binmethod = BinSimple()
binmethod = BinMRD()
# med_obs=[200,20,20]
# photerr = GaiaDR3(model, med_obs)
photerr = Individual(model, photsys, observation)
isoc_inst = Isoc(parsec_inst)
synstars_inst = SynStars(model, photsys,
                         imf_inst, binmethod, photerr)

# %%
logage, mh, dm, Av, fb = 9.5, 0.05, 9.6, 0.16, 0.3  # NGC2682
# logage, mh, dm, Av, fb = 8.6, 0.0, 8.45, 0.18, 0.27  # NGC 3532
# logage, mh, dm, Av, fb = 10.0, -1.8, 10.0, 0.16, 0.3
theta = logage, mh, dm, Av, fb
n_stars = 1500

samples, _, _, _, _, _ = synstars_inst(theta, n_stars, isoc_inst, test=True, figure=True)
observation = samples.rename(columns={'G': 'Gmag', 'BP': 'BPmag', 'RP': 'RPmag'})
# %%
bins = 50
h2h_cmd_inst = Hist2Hist4CMD(model, photsys, bins)
h2p_cmd_inst = Hist2Point4CMD(model, photsys, bins)

# logage, mh, dm, Av, fb = 10.0, -1.8, 10.0, 0.16, 0.3
# logage, mh, dm, Av, fb = 8.6, 0.0, 8.45, 0.18, 0.27  # NGC3532
logage, mh, dm, Av, fb = 9.5, 0.05, 9.6, 0.16, 0.3  # NGC2682
theta = logage, mh, dm, Av, fb
step = (0.05, 0.05)  # logage, mh
n_stars = 10000

samples1, accepted_rate, total_size, test_sample_time, isoc, isoc_new = synstars_inst(
    theta, n_stars, isoc_inst, test=True, figure=True)
# %%
h2h = lnlike_5p(theta, step, isoc_inst, h2h_cmd_inst, synstars_inst, n_stars, observation, 5)
h2p = lnlike_5p(theta, step, isoc_inst, h2p_cmd_inst, synstars_inst, n_stars, observation, 5)
print(h2h)
print(h2p)

fig, ax = plt.subplots(figsize=(4, 4))
bin1 = ~samples1['mass_sec'].isna()
ax.scatter(samples1['BP'][bin1] - samples1['RP'][bin1], samples1['G'][bin1], color='blue', s=5, alpha=0.6)
ax.scatter(samples1['BP'][~bin1] - samples1['RP'][~bin1], samples1['G'][~bin1], color='orange', s=5, alpha=0.6)
ax.scatter(observation['BPmag'] - observation['RPmag'], observation['Gmag'], s=1, color='gray')
ax.invert_yaxis()
ax.set_xlim(min(observation['BPmag'] - observation['RPmag']) - 0.2,
            max(observation['BPmag'] - observation['RPmag']) + 0.2)
ax.set_ylim(max(observation['Gmag']) + 0.5, min(observation['Gmag']) - 0.5)
ax.set_title(f'logage={logage}, [M/H]={mh},\n'
             f'DM={dm}, Av={Av}, fb={fb}', fontsize=12)
ax.text(0.6, 0.9, f'H2H={h2h:.4f}', transform=ax.transAxes, fontsize=11)
ax.text(0.6, 0.8, f'H2P={h2p:.4f}', transform=ax.transAxes, fontsize=11)
fig.show()

# %%
# p0以初值为均值分布
ndim = 5
nwalkers = 50

# scale = np.array([1, 0.1, 10, 0.5, 0.1])
# p0 = theta + scale * np.random.randn(nwalkers, ndim)
labels = ['log(age)', '[M/H]', 'DM', '$A_{v}$', '$f_{b}$']
temp = []
scale = np.array([1, 0.1, 5, 0.5, 0.1])
theta_range = [[6.7, 10.0], [-2.0, 0.4], [3.0, 15.0], [0.0, 3.0], [0.2, 1.0]]

for i in range(ndim):
    aux_list = []
    while len(aux_list) < nwalkers:
        aux = theta[i] + scale[i] * np.random.randn()
        if theta_range[i][0] <= aux <= theta_range[i][1]:
            aux_list.append(aux)
    plt.scatter(aux_list, np.full(nwalkers, 10))
    plt.xlabel(labels[i])
    plt.show()
    temp.append(aux_list)
p0 = np.array(temp).T

# %%
# p0在参数空间内均匀分布
ndim = 5
nwalkers = 50
theta_range = [[6.7, 10.0], [-2.0, 0.4], [3.0, 15.0], [0.0, 3.0], [0.2, 1.0]]

theta_samples = []
for bounds in theta_range:
    lower_bound, upper_bound = bounds
    samples = np.linspace(lower_bound, upper_bound, nwalkers)
    np.random.shuffle(samples)
    theta_samples.append(samples)
p0 = np.array(theta_samples).T

# %%
likelihood_inst = h2p_cmd_inst
with Pool(8) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnlike_5p,
        pool=pool,
        args=(step, isoc_inst, likelihood_inst, synstars_inst, n_stars, observation, 5)
    )
    nburn = 500
    p1, prob, state = sampler.run_mcmc(p0, nburn, progress=True)

    # sampler.reset()
    # pos, prob, state = sampler.run_mcmc(p1, 2000, progress=True)


# %%
# # 获取采样样本链和对应的 ln(probability)
# samples = sampler.flatchain
# ln_prob = sampler.flatlnprobability
# # 找到具有最大 ln(probability) 的索引
# max_prob_index = np.argmax(ln_prob)
# # 从样本链中获取最大 ln(probability) 对应的参数值
# max_prob_sample = samples[max_prob_index]

fig = corner.corner(
    sampler.flatchain,
    labels=['log(age)', '[M/H]', 'DM', '$A_{v}$', '$f_{b}$'],
    truths=[9.5, 0.05, 9.6, 0.16, 0.3],  # NGC2682
    # truths=[8.6, 0.0, 8.45, 0.18, 0.27],  # NGC3532
    # truths=[10.0, -1.8, 10.0, 0.16, 0.3],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={'fontsize': 18},
    # title_fmt='.2f'
)
fig.show()

labels = ['log(age)', '[M/H]', 'DM', '$A_{v}$', '$f_{b}$']
samples = sampler.get_chain()
fig, axes = plt.subplots(ndim, figsize=(10, 8), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i])
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel('step number')
fig.show()
