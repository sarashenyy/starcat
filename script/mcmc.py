from multiprocessing import Pool

import corner
import emcee
import joblib
import numpy as np

from script.widgets import read_sample_obs
from starcat import (Isoc, IMF, Parsec,
                     GaiaEDR3, BinSimple, Hist2Hist,
                     lnlike_2p, SynStars)

sample_obs, med_nobs = read_sample_obs('melotte_22_edr3', 'gaiaEDR3')
# !NOTE Gmag range
sample_obs = sample_obs[sample_obs['Gmag'] < 10]
# CMD.plot_cmd(sample_obs, 'parsec','gaiaEDR3',20)

# parameter
theta_2p = (7.89, 0.032)
fb, dm = 0.35, 5.55
step = (0.01, 0.01)
n_stars = 10000
bins = 20

# initial methos
parsec = Parsec()
isoc = Isoc(parsec)
imf = IMF('kroupa01')
# !CHECK binmethod
binmethod = BinSimple()
photerr = GaiaEDR3('parsec', med_nobs)
# !CHECK likelihoodfunc
likelihoodfunc = Hist2Hist('parsec', 'gaiaEDR3', bins)
synstars = SynStars('parsec', 'gaiaEDR3', imf, n_stars, binmethod, photerr)

# lnlike_2p(theta_2p, fb, dm, step, isoc, likelihoodfunc, synstars, sample_obs)

# MCMC
ndim = 2
# set up MCMC sampler
nwalkers = 20
# scale = np.array([0.1, 0.1])
# p0 = np.round((theta_2p + scale * np.random.randn(nwalkers, ndim)), 2)
scale = np.array([0.1, 0.01])
p0 = theta_2p + scale * np.random.randn(nwalkers, ndim)
# parallelzation
with Pool() as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnlike_2p,
        args=(fb, dm, step, isoc, likelihoodfunc, synstars, sample_obs)
        # moves=[
        #     (emcee.moves.DEMove(), 0.8),
        #     (emcee.moves.DESnookerMove(), 0.2),
        # ]
    )
    nburn = 1000
    pos, _, _, = sampler.run_mcmc(p0, nburn, progress=True)

    sampler.reset()
    nsteps = 2000
    sampler.run_mcmc(pos, nsteps, progress=True)

joblib.dump(sampler.flatchain, '/home/shenyueyue/Projects/starcat/test_data/f.joblib')
joblib.dump(sampler.get_chain(), '/home/shenyueyue/Projects/starcat/test_data/s.joblib')

# corner
fig = corner.corner(
    sampler.flatchain,
    labels=['log(age)', '[M/H]'],
    truths=[7.89, 0.032],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={'fontsize': 18},
    title_fmt='.2f'
)
fig.show()
fig.savefig('/home/shenyueyue/Projects/starcat/test_data/corner.png')

import matplotlib.pyplot as plt

labels = ['log(age)', '[M/H]']
samples = sampler.get_chain()
fig, axes = plt.subplots(ndim, figsize=(10, 4), sharex=True)
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i])
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel('step number')
fig.show()
fig.savefig('/home/shenyueyue/Projects/starcat/test_data/chain.png')
# ax.set_xlim(50, 2000)
