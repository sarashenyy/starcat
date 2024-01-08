from multiprocessing import Pool

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import gridspec

from starcat import (Isoc, IMF, MIST,
                     BinMRD, Individual,
                     SynStars,
                     lnlike, Hist2Hist4CMD, Hist2Point4CMD)

# %matplotlib osx
plt.style.use('/Users/sara/PycharmProjects/starcat/data/mystyle.mplstyle')
# %%
file = '/Users/sara/PycharmProjects/starcat/data/MCdata/NGC1866_data_new.csv'
cluster_name = 'NGC 1866'
center_ra = '05h13m38.7s'
center_dec = '-65d27m52.15s'
data = pd.read_csv(file)

data = data[(data['F814W'] < 21.0) & (data['F336W'] < 22.0)
            & (data['F336W_err'] < np.std(data['F336W_err']))
            & (data['F814W_err'] < np.std(data['F814W_err']))]
# data = data[data['F814W'] < 21.0]

data_coords = SkyCoord(data['RA'] * u.deg, data['DEC'] * u.deg)
center = SkyCoord(center_ra, center_dec, frame='icrs')
angular_distance = data_coords.separation(center)
r_half_mass = 13.92 / 3600 * u.deg

n = 1
r = r_half_mass * n
field = angular_distance > r

# tt = np.sqrt(((data['RA']-center.ra.deg)*np.cos(data['DEC']))**2 + (data['DEC']-center.dec.deg)**2)
# tt - angular_distance.value

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2)
gs.update(wspace=0.3, hspace=0.4)

# draw cluater area
ra = data['RA']
dec = data['DEC']
ax_loc = plt.subplot(gs[0, 0])
ax_loc.scatter(ra[~field], dec[~field], s=3, c='k')
ax_loc.scatter(ra[field], dec[field], s=3, c='grey', label='field')
ax_loc.scatter(center.ra, center.dec, c='r', marker='+')
ax_loc.set_title(f'{n}' + '$R_{h}$')
ax_loc.set_xlabel('RA')
ax_loc.set_ylabel('DEC')

# draw total CMD
c = data['F336W'] - data['F814W']
m = data['F814W']
ax_cmd = plt.subplot(gs[0, 1])
ax_cmd.scatter(c[~field], m[~field], s=3, c='k')
ax_cmd.scatter(c[field], m[field], s=3, c='grey')
ax_cmd.invert_yaxis()
ax_cmd.set_xlabel('F336W - F814W')
ax_cmd.set_ylabel('F814W')

# draw cluster CMD
ax_cluster = plt.subplot(gs[1, 0])
ax_cluster.scatter(c[~field], m[~field], s=3, c='k')
ax_cluster.invert_yaxis()
ax_cluster.set_title(cluster_name)
ax_cluster.set_xlabel('F336W - F814W')
ax_cluster.set_ylabel('F814W')

# draw field CMD
ax_field = plt.subplot(gs[1, 1])
ax_field.scatter(c[field], m[field], s=3, c='grey')
ax_field.invert_yaxis()
ax_field.set_title('field')
ax_field.set_xlabel('F336W - F814W')
ax_field.set_ylabel('F814W')

fig.show()
observation = data[~field]
# %%
# initialization
photsys = 'HSTWFC3'
model = 'mist'
imf = 'kroupa01'
imf_inst = IMF(imf)
mist_inst = MIST()
binmethod = BinMRD()
photerr = Individual(model, photsys, observation)
isoc_inst = Isoc(mist_inst)
synstars_inst = SynStars(model, photsys,
                         imf_inst, binmethod, photerr)

# %%
logage, mh, dm, Av, fb, alpha = 8.35, -0.5, 18.2, 0.45, 0.3, 2.35  # 7.65, -0.4, 18.58, 0.09, 0.3, 2.35
theta = logage, mh, dm, Av, fb, alpha
n_stars = 1000

isoc_obs = pd.DataFrame([])
aux = isoc_inst.get_isoc('HSTWFC3', logage=logage, mh=mh)
sys_param = {'F336W': [3358.61, 512.0, 1.67521],
             'F814W': [8058.20, 1541.0, 0.59918]}
bands = ['F336W', 'F814W']
mass = 'initial_mass'
isoc_obs[mass] = aux[mass]
for j, band in enumerate(bands):
    _, _, c = sys_param[band]
    isoc_obs[band] = aux[band] + dm + c * Av

samples = synstars_inst(theta, n_stars, isoc_inst)
observation = samples  # samples[(samples['g'] < 26.0)]  # samples[(samples['g'] < 25.5) & (samples['i'] < 25.5)]

# %%
# %matplotlib osx
fig = plt.figure(figsize=(10, 4.5))
gs = gridspec.GridSpec(1, 2)
gs.update(wspace=0.2)

ax = plt.subplot(gs[0, 0])
ax.scatter(observation['F336W'] - observation['F814W'], observation['F814W'], s=5)
ax.plot(isoc_obs['F336W'] - isoc_obs['F814W'], isoc_obs['F814W'], c='r', linewidth=1)
ax.set_ylim(max(observation['F814W']) + 0.2, min(observation['F814W']) - 0.5)
ax.set_xlim(min(observation['F336W'] - observation['F814W']) - 0.2,
            max(observation['F336W'] - observation['F814W']) + 0.2)
ax.set_title(f'logage={logage}, [M/H]={mh}, \n'
             f'DM={dm}, Av={Av}, fb={fb}, alpha={alpha}')
# ax.set_title('observation')
ax.set_xlabel('F336W - F814W')
ax.set_ylabel('F814W')

ax2 = plt.subplot(gs[0, 1])
ax2.scatter(samples['F336W'] - samples['F814W'], samples['F814W'], s=5)
ax2.plot(isoc_obs['F336W'] - isoc_obs['F814W'], isoc_obs['F814W'], c='r', linewidth=1)
ax2.set_xlim(ax.get_xlim())
ax2.set_ylim(ax.get_ylim())
ax2.set_title('uniform q')
ax2.set_xlabel('F336W - F814W')

fig.show()
# %%
bins = 50
h2h_cmd_inst = Hist2Hist4CMD(model, photsys, bins)
h2p_cmd_inst = Hist2Point4CMD(model, photsys, bins)

logage, mh, dm, Av, fb, alpha = 8.35, -0.5, 18.2, 0.45, 0.3, 2.35
theta = logage, mh, dm, Av, fb, alpha
step = (0.05, 0.5)  # logage, mh
n_stars = 10000

samples = synstars_inst(theta, n_stars, isoc_inst)

h2h = lnlike(theta, step, isoc_inst, h2h_cmd_inst, synstars_inst, n_stars, observation, 'LG', 5)
h2p = lnlike(theta, step, isoc_inst, h2p_cmd_inst, synstars_inst, n_stars, observation, 'LG', 5)
print(h2h)
print(h2p)

fig, ax = plt.subplots(figsize=(4, 4))
bin = samples['mass_sec'].notna()

c = samples['F336W'] - samples['F814W']
m = samples['F814W']
c_obs = observation['F336W'] - observation['F814W']
m_obs = observation['F814W']
ax.scatter(c[bin], m[bin], color='blue', s=5, alpha=0.6)
ax.scatter(c[~bin], m[~bin], color='orange', s=5, alpha=0.6)
ax.scatter(c_obs, m_obs, s=1, color='grey')
ax.invert_yaxis()
ax.set_xlim(min(c_obs) - 0.2, max(c_obs) + 0.2)
ax.set_ylim(max(m_obs) + 0.5, min(m_obs) - 0.5)
ax.set_title(f'logage={logage}, [M/H]={mh},\n'
             f'DM={dm}, Av={Av}, fb={fb}, alpha={alpha}', fontsize=12)
ax.text(0.6, 0.3, f'H2H={h2h:.2f}', transform=ax.transAxes, fontsize=11)
ax.text(0.6, 0.2, f'H2P={h2p:.2f}', transform=ax.transAxes, fontsize=11)
fig.show()

# %%
# p0以真值为均值
ndim = 6
nwalkers = 60

# scale = np.array([1, 0.1, 10, 0.5, 0.1])
# p0 = theta + scale * np.random.randn(nwalkers, ndim)
labels = ['log(age)', '[M/H]', 'DM', '$A_{v}$', '$f_{b}$', 'alpha']
temp = []
scale = np.array([0.5, 0.1, 1, 0.5, 0.1, 0.2])
theta_range = [[6.7, 10.0], [-2.0, 0.4],
               [15.0, 19.0], [0.0, 2.0],
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
likelihood_inst = h2p_cmd_inst
with Pool(10) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnlike,
        pool=pool,
        args=(step, isoc_inst, likelihood_inst, synstars_inst, n_stars, observation, 'LG', 5)
    )
    nburn = 2000
    pos, prob, state = sampler.run_mcmc(p0, nburn, progress=True)

    # sampler.reset()
    # pos, prob, state = sampler.run_mcmc(p1, 10000, progress=True)

# %%
# 获取采样样本链和对应的 ln(probability)
# tau = sampler.get_autocorr_time()
# print(tau)

samples = sampler.flatchain
ln_prob = sampler.flatlnprobability
# 找到具有最大 ln(probability) 的索引
max_prob_index = np.argmax(ln_prob)
# 从样本链中获取最大 ln(probability) 对应的参数值
max_prob_sample = samples[max_prob_index]

fig = corner.corner(
    # sampler.get_chain(discard=1000, thin=10, flat=True),
    samples,
    labels=['log(age)', '[M/H]', 'DM', '$A_{v}$', '$f_{b}$', '$\\alpha$'],
    truths=[8.35, -0.5, 18.2, 0.45, 0.3, 2.35],
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
truths = ((8.35, -0.5, 18.2, 0.45, 0.3, 2.35),
          (8.34, -0.49, 18.2, 0.45, 0.35, 2.47))
paramRanges = ((8.0, 8.8), (-0.7, -0.3), (17.6, 19), (0., 1.0), (0.2, 0.5), (2.0, 2.8))
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
title = ['log(age)=${8.34}_{-0.06}^{+0.01}$',
         '     [M/H]=${-0.49}_{-0.04}^{+0.04}$',
         'DM=${18.20}_{-0.00}^{+0.09}$',
         '$A_{v}$=${0.45}_{-0.00}^{+0.07}$',
         '$f_{b}$=${0.35}_{-0.02}^{+0.02}$',
         '$\\alpha$=${2.47}_{-0.20}^{+0.07}$']
for i in range(ndim):
    j = len(axes) - ndim + i
    ax = axes[j]
    ax.set_title(f'   {title[i]}', fontsize=8)
# GTC.show()
GTC.savefig('fullGTC_ngc1866.pdf', bbox_inches='tight')
