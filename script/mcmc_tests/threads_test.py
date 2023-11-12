import emcee
import joblib
import numpy as np

from starcat import (config,
                     Isoc, IMF, Parsec, CSSTsim,
                     BinSimple, SynStars,
                     lnlike_5p, Hist2Hist4CMD, Hist2Point4CMD)

dir = '/home/shenyueyue/Projects/starcat/data/mcmc/threads_test/'
logage_val = 7.
sample_val_path = dir + f'sample_val{int(logage_val)}.joblib'

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
binmethod = BinSimple()
photerr = CSSTsim(model)
# ?init SynStars
synstars_inst = SynStars(model, photsys, imf_inst, binmethod, photerr)
# ?init LikelihoodFunc
h2h_cmd_inst = Hist2Hist4CMD(model, photsys, bins)
h2p_cmd_inst = Hist2Point4CMD(model, photsys, bins)

# !create synthetic cluster for validation
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

# %%

# MCMC
ndim = 5
# set up MCMC sampler
nwalkers = 20
# scale = np.array([0.1, 0.1])
# p0 = np.round((theta_2p + scale * np.random.randn(nwalkers, ndim)), 2)
scale = np.array([1, 0.5, 50, 1, 0.1])
p0 = theta_val + scale * np.random.randn(nwalkers, ndim)

logage_step = 0.01  # 0.1
mh_step = 0.01  # 0.05
step = (logage_step, mh_step)

# with Pool(20) as pool:
# sampler = emcee.EnsembleSampler(
#     nwalkers, ndim, lnlike_5p,
#     args=(step, isoc_inst, h2h_cmd_inst, synstars_inst, n_stars, sample_val),
#     # moves=[
#     #     (emcee.moves.DEMove(), 0.8),
#     #     (emcee.moves.DESnookerMove(), 0.2),
#     # ]
# )
# nburn = 2000
# pos, _, _, = sampler.run_mcmc(p0, nburn, progress=True)

from joblib import Parallel, delayed

n_steps = 2000  # 总步数


# 创建一个函数用于运行 MCMC
def run_mcmc_chain(p0, n_steps):
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnlike_5p,
        args=(step, isoc_inst, h2h_cmd_inst, synstars_inst, n_stars, sample_val),
    )
    sampler.run_mcmc(p0, n_steps, progress=True)
    return sampler.get_chain()


# 并行运行 MCMC 链
n_chains = 20  # 指定要并行运行的链的数量
chains = Parallel(n_jobs=n_chains)(delayed(run_mcmc_chain)(p0, n_steps) for _ in range(n_chains))

# 合并并分析结果
combined_samples = np.vstack([chain for chain in chains])
# 进行采样结果的分析和可视化

# %%
# with Pool() as pool:
#     sampler.reset()
#     nsteps = 2000
#     sampler.run_mcmc(pos, nsteps, progress=True)

# joblib.dump(sampler.flatchain, dir + 'flatchain.joblib')
# joblib.dump(sampler.get_chain(), dir + 'chain.joblib')
joblib.dump(combined_samples, dir + 'flatchain.joblib')
# corner
# fig = corner.corner(
#     sampler.flatchain,
#     labels=['log(age)', '[M/H]'],
#     truths=[9.57, 0.045],
#     quantiles=[0.16, 0.5, 0.84],
#     show_titles=True,
#     title_kwargs={'fontsize': 18},
#     title_fmt='.2f'
# )
# fig.show()
# fig.savefig('/home/shenyueyue/Projects/starcat/test_data/corner_move.png')
#
# import matplotlib.pyplot as plt
#
# labels = ['log(age)', '[M/H]']
# samples = sampler.get_chain()
# fig, axes = plt.subplots(ndim, figsize=(10, 4), sharex=True)
# for i in range(ndim):
#     ax = axes[i]
#     ax.plot(samples[:, :, i])
#     ax.set_ylabel(labels[i])
# axes[-1].set_xlabel('step number')
# fig.show()
# fig.savefig('/home/shenyueyue/Projects/starcat/test_data/chain_move.png')
# ax.set_xlim(50, 2000)
