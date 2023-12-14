from multiprocessing import Pool

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np

from starcat import (Isoc, Parsec, IMF,
                     BinSimple,
                     CSSTsim, SynStars,
                     lnlike_5p, Hist2Hist4CMD, Hist2Point4CMD)

# %matplotlib osx
plt.style.use('/Users/sara/PycharmProjects/starcat/data/mystyle.mplstyle')
# %%
# initialization
photsys = 'CSST'
model = 'parsec'
imf = 'kroupa01'
imf_inst = IMF(imf)
parsec_inst = Parsec()
binmethod = BinSimple()
photerr = CSSTsim(model)
isoc_inst = Isoc(parsec_inst)
synstars_inst = SynStars(model, photsys,
                         imf_inst, binmethod, photerr)

# %%
logage, mh, dm, Av, fb = 8.4, 0.0, 18.5, 0.4, 0.5
theta = logage, mh, dm, Av, fb
n_stars = 1500

samples = synstars_inst(theta, n_stars, isoc_inst)
observation = samples  # samples[(samples['g'] < 26.0)]  # samples[(samples['g'] < 25.5) & (samples['i'] < 25.5)]

fig, ax = plt.subplots(figsize=(4, 5))
ax.scatter(observation['g'] - observation['i'], observation['i'], s=3, label='mag=i', alpha=0.5)
ax.scatter(observation['g'] - observation['i'], observation['g'], s=3, label='mag=g', alpha=0.5)
ax.set_ylim(max(observation['g']) + 0.2, min(observation['g']) - 0.5)
ax.set_xlim(min(observation['g'] - observation['i']) - 0.2, max(observation['g'] - observation['i']) + 0.2)
ax.legend()
ax.set_title(f'logage={logage}, [M/H]={mh}, \n'
             f'DM={dm}, Av={Av}, fb={fb}')
ax.set_xlabel('g-i')
ax.set_ylabel('mag')
fig.show()
# %%
bins = 50
h2h_cmd_inst = Hist2Hist4CMD(model, photsys, bins)
h2p_cmd_inst = Hist2Point4CMD(model, photsys, bins)

logage, mh, dm, Av, fb = 8.4, 0.0, 18.5, 0.4, 0.5
theta = logage, mh, dm, Av, fb
step = (0.05, 0.05)  # logage, mh
n_stars = 10000

print(lnlike_5p(theta, step, isoc_inst, h2h_cmd_inst, synstars_inst, n_stars, observation, 'LG', 5))
print(lnlike_5p(theta, step, isoc_inst, h2p_cmd_inst, synstars_inst, n_stars, observation, 'LG', 5))

# %%
# p0以真值为均值
ndim = 5
nwalkers = 50

# scale = np.array([1, 0.1, 10, 0.5, 0.1])
# p0 = theta + scale * np.random.randn(nwalkers, ndim)
labels = ['log(age)', '[M/H]', 'DM', '$A_{v}$', '$f_{b}$']
temp = []
scale = np.array([1, 0.1, 5, 0.5, 0.1])
theta_range = [[6.7, 10.0], [-2.0, 0.4], [20.0, 28.0], [0.0, 3.0], [0.2, 1.0]]

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
# p0在参数范围内均匀分布
ndim = 5
nwalkers = 50
theta_range = [[6.7, 10.0], [-2.0, 0.4], [20.0, 28.0], [0.0, 3.0], [0.2, 1.0]]

theta_samples = []
for bounds in theta_range:
    lower_bound, upper_bound = bounds
    samples = np.linspace(lower_bound, upper_bound, nwalkers)
    np.random.shuffle(samples)
    theta_samples.append(samples)
p0 = np.array(theta_samples).T

# %%
likelihood_inst = h2p_cmd_inst
with Pool(10) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnlike_5p,
        pool=pool,
        args=(step, isoc_inst, likelihood_inst, synstars_inst, n_stars, observation, 'LG', 5)
    )
    nburn = 3500
    pos, prob, state = sampler.run_mcmc(p0, nburn, progress=True)

    # sampler.reset()
    # pos, prob, state = sampler.run_mcmc(p1, 10000, progress=True)

# %%


# 获取采样样本链和对应的 ln(probability)
samples = sampler.flatchain
ln_prob = sampler.flatlnprobability
# 找到具有最大 ln(probability) 的索引
max_prob_index = np.argmax(ln_prob)
# 从样本链中获取最大 ln(probability) 对应的参数值
max_prob_sample = samples[max_prob_index]

fig = corner.corner(
    sampler.flatchain,
    labels=['log(age)', '[M/H]', 'DM', '$A_{v}$', '$f_{b}$'],
    truths=theta,
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
