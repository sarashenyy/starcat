from multiprocessing import Pool

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np

from starcat import (Isoc, Parsec, IMF,
                     BinMRD,
                     CSSTsim, SynStars,
                     lnlike, Hist2Hist4CMD, Hist2Point4CMD)

# %matplotlib osx
# plt.style.use('/Users/sara/PycharmProjects/starcat/data/mystyle.mplstyle')
# plt.style.use('/home/shenyueyue/Projects/starcat/data/mystyle.mplstyle')
# %%
# initialization
photsys = 'CSST'
model = 'parsec'
imf = 'salpeter55'
imf_inst = IMF(imf)
parsec_inst = Parsec()
binmethod = BinMRD()
photerr = CSSTsim(model)
isoc_inst = Isoc(parsec_inst)
synstars_inst = SynStars(model, photsys,
                         imf_inst, binmethod, photerr)

# %%
logage, mh, dm, Av, fb, alpha = 8.4, 0.0, 18.5, 0.4, 0.5, 2.35
theta = logage, mh, dm, Av, fb, alpha
n_stars = 1500

samples = synstars_inst(theta, n_stars, isoc_inst)
observation = samples  # samples[(samples['g'] < 26.0)]  # samples[(samples['g'] < 25.5) & (samples['i'] < 25.5)]

fig, ax = plt.subplots(figsize=(4, 5))
bin = observation['mass_sec'].notna()
c = observation['g'] - observation['i']
m = observation['i']

ax.scatter(c[bin], m[bin], color='blue', s=5, alpha=0.6, label='binary')
ax.scatter(c[~bin], m[~bin], color='orange', s=5, alpha=0.6, label='single')
ax.set_ylim(max(observation['i']) + 0.2, min(observation['i']) - 0.5)
ax.set_xlim(min(observation['g'] - observation['i']) - 0.2, max(observation['g'] - observation['i']) + 0.2)
ax.legend()
ax.set_title(f'logage={logage}, [M/H]={mh}, DM={dm}, \n'
             f'Av={Av}, fb={fb}, alpha={alpha}', fontsize=12)
ax.set_xlabel('g - i')
ax.set_ylabel('i')

fig.show()
# %%
bins = 50
h2h_cmd_inst = Hist2Hist4CMD(model, photsys, bins)
h2p_cmd_inst = Hist2Point4CMD(model, photsys, bins)

logage, mh, dm, Av, fb, alpha = 8.4, 0.0, 18.5, 0.4, 0.5, 2.35
theta = logage, mh, dm, Av, fb, alpha
step = (0.05, 0.05)  # logage, mh
n_syn = 10000

samples = synstars_inst(theta, n_syn, isoc_inst)

h2h = lnlike(theta, step, isoc_inst, h2h_cmd_inst, synstars_inst, n_syn, observation, 'LG', 5)
h2p = lnlike(theta, step, isoc_inst, h2p_cmd_inst, synstars_inst, n_syn, observation, 'LG', 5)
print(h2h)
print(h2p)

fig, ax = plt.subplots(figsize=(4, 4))
bin = samples['mass_sec'].notna()
c = samples['g'] - samples['i']
m = samples['i']
c_obs = observation['g'] - observation['i']
m_obs = observation['i']

ax.scatter(c[bin], m[bin], color='blue', s=5, alpha=0.6)
ax.scatter(c[~bin], m[~bin], color='orange', s=5, alpha=0.6)
ax.scatter(c_obs, m_obs, s=1, color='grey')
ax.invert_yaxis()
ax.set_xlim(min(c_obs) - 0.2, max(c_obs) + 0.2)
ax.set_ylim(max(m_obs) + 0.5, min(m_obs) - 0.5)
ax.set_title(f'logage={logage}, [M/H]={mh}, DM={dm}, \n'
             f'Av={Av}, fb={fb}, alpha={alpha}', fontsize=12)
ax.set_xlabel('g - i')
ax.set_ylabel('i')
ax.text(0.6, 0.6, f'H2H={h2h:.2f}', transform=ax.transAxes, fontsize=11)
ax.text(0.6, 0.5, f'H2P={h2p:.2f}', transform=ax.transAxes, fontsize=11)

fig.show()

# %%
# p0以真值为均值
ndim = 6
nwalkers = 30

# scale = np.array([1, 0.1, 10, 0.5, 0.1])
# p0 = theta + scale * np.random.randn(nwalkers, ndim)
labels = ['log(age)', '[M/H]', 'DM', '$A_{v}$', '$f_{b}$', 'alpha']
temp = []
scale = np.array([0.5, 0.1, 1, 0.5, 0.1, 0.2])
theta_range = [[6.7, 10.0], [-2.0, 0.4],
               [17.5, 25.5], [0.0, 2.0],
               [0.2, 1.0], [1.6, 3.0]]

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
with Pool(10) as pool:  # local:10(20min) ; raku:30(20min) ;
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnlike,
        pool=pool,
        args=(step, isoc_inst, likelihood_inst, synstars_inst, n_syn, observation, 'LG', 5)
    )
    # nburn = 500
    nstep = 2500
    # p1, _, _ = sampler.run_mcmc(p0, nburn, progress=True)
    # sampler.reset()
    pos, prob, state = sampler.run_mcmc(p0, nstep, progress=True)

# %%
# test Autocorrelation Time
# https://emcee.readthedocs.io/en/stable/tutorials/monitor/?highlight=sampler.sample
# The difference here was the addition of a “backend”.
# This choice will save the samples to a file called tutorial.h5 in the current directory.
# Now, we’ll run the chain for up to 10,000 steps and check the autocorrelation time every 100 steps.
# If the chain is longer than 100 times the estimated autocorrelation time and if this estimate changed by less than 1%,
# we’ll consider things converged.
coords = p0
likelihood_inst = h2p_cmd_inst
# Set up the backend
# Don't forget to clear it in case the file already exists
filename = "tutorial.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

# Initialize the sampler
with Pool(20) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnlike,
        pool=pool,
        args=(step, isoc_inst, likelihood_inst, synstars_inst, n_syn, observation, 'LG', 5),
        backend=backend)

    max_n = 100000

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(coords, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau

# %%
import matplotlib.pyplot as plt

n = 100 * np.arange(1, index + 1)
y = autocorr[:index]
plt.plot(n, n / 100.0, "--k")
plt.plot(n, y)
plt.xlim(0, n.max())
plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
plt.xlabel("number of steps")
plt.ylabel(r"mean $\hat{\tau}$");
plt.show()

# %%
# 获取采样样本链和对应的 ln(probability)
samples = sampler.flatchain
ln_prob = sampler.flatlnprobability
# 找到具有最大 ln(probability) 的索引
max_prob_index = np.argmax(ln_prob)
# 从样本链中获取最大 ln(probability) 对应的参数值
max_prob_sample = samples[max_prob_index]

fig = corner.corner(
    # sampler.get_chain(discard=1000, thin=10, flat=True),
    sampler.get_chain()[5000:, :, :].reshape((-1, ndim)),
    # samples,
    labels=['log(age)', '[M/H]', 'DM', '$A_{v}$', '$f_{b}$', '$\\alpha$'],
    truths=list(theta),
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={'fontsize': 18},
    # title_fmt='.2f'
)
fig.show()

# Extract the axes
axes = np.array(fig.axes).reshape((ndim, ndim))

# Loop over the diagonal
title = []
for i in range(ndim):
    ax = axes[i, i]
    aux = ax.get_title()
    title.append(aux)

labels = ['log(age)', '[M/H]', 'DM', '$A_{v}$', '$f_{b}$', '$\\alpha$']
samples = sampler.get_chain()
# samples = sampler.get_chain()[500:, :, :]
fig, axes = plt.subplots(ndim, figsize=(10, 8), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i])
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel('step number')
fig.show()

# %%
import pygtc

plt.style.use('default')
truths = list(theta)
paramRanges = ((7.5, 9.3), (-0.3, 0.15), (17.5, 19.5), (0.2, 0.8), (0.4, 0.9), (2.0, 2.6))
paramNames = ('log(age)', '[M/H]', 'DM', '$A_{v}$', '$f_{b}$', '$\\alpha$')
truthColors = ('#FF8000',
               '#1F77B4')  # , '#FF8000', '#FF8000'
GTC = pygtc.plotGTC(chains=sampler.flatchain,
                    paramNames=paramNames,
                    truths=truths,
                    nContourLevels=2,
                    # sigmaContourLevels=True,
                    paramRanges=paramRanges,
                    truthColors=truthColors,
                    nBins=20,
                    figureSize='MNRAS_page')

# Extract the axes
axes = np.array(GTC.axes)

# check axes index
# for i in range(len(axes)):
#     ax = axes[i]
#     ax.text(0.5, 0.5, f'{i}', transform=ax.transAxes)
# Loop over the diagonal
# title=['${8.34}_{-0.06}^{+0.01}$',
#  '${-0.49}_{-0.04}^{+0.04}$',
#  '${18.20}_{-0.00}^{+0.09}$',
#  '${0.45}_{-0.00}^{+0.07}$',
#  '${0.35}_{-0.02}^{+0.02}$',
#  '${2.47}_{-0.20}^{+0.07}$']

for i in range(ndim):
    j = len(axes) - ndim + i
    ax = axes[j]
    ax.set_title(f'   {title[i]}', fontsize=8)
GTC.show()
# GTC.savefig('fullGTC_ngc1866.pdf', bbox_inches='tight')

# %%
# 初始化一个空数组来存储结果
result = None

for i in range(ndim):
    temp = corner.quantile(sampler.flatchain[:, i], [0.16, 0.5, 0.84])
    if result is None:
        result = temp
    else:
        result = np.vstack((result, temp))
