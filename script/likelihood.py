import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from script.widgets import read_sample_obs
from starcat import (GaiaEDR3, Parsec,
                     BinSimple, Hist2Point4CMD,
                     Hist2Hist4CMD, Isoc, IMF, SynStars, lnlike_2p)
from starcat.test import likelihood_test


def test_randomness(theta, n_stars, step, times, likelihoodfunc):
    # ? input: likelihoodfunc = Hist2Point('parsec', 'gaiaEDR3', 50)

    sample_obs3, med_nobs3 = read_sample_obs('melotte_22_edr3', 'gaiaEDR3')
    # !NOTE Gmag<10
    sample_obs3 = sample_obs3[sample_obs3['Gmag'] < 10]
    logage, mh, fb, dm = theta
    theta_2p = logage, mh
    print(f"calculate lnlike values in total : {times} times")
    parsec = Parsec()
    isoc = Isoc(parsec)
    imf = IMF('chabrier03')
    binmethod = BinSimple()

    photerr3 = GaiaEDR3('parsec', med_nobs3)

    synstars3 = SynStars('parsec', 'gaiaEDR3', imf, n_stars, binmethod, photerr3)

    sample_syn = synstars3(theta, step, isoc)
    sample_syn = sample_syn[['Gmag', 'G_BPmag', 'G_RPmag']]
    lnlike_list = []
    for i in range(times):
        lnlike = lnlike_2p(theta_2p, fb, dm, step, isoc, likelihoodfunc, synstars3, sample_obs3)
        lnlike_list.append(lnlike)
        if i % 100 == 0:
            print(i)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(lnlike_list, bins=50)
    ax.set_xlabel('lnlikelihood')
    ax.text(0.7, 0.95, f"mean: {np.mean(lnlike_list):.1f}", transform=ax.transAxes)
    ax.text(0.7, 0.90, f"std: {np.std(lnlike_list):.1f}", transform=ax.transAxes)
    ax.set_title(f"logage:{logage} [M/H]:{mh} fb:{fb} dm:{dm} nstars:{n_stars}")
    fig.show()

    return lnlike_list


theta1 = 7.9, 0.03, 0.35, 5.55
theta2 = 7.95, 0.03, 0.35, 5.55
n_stars = 10000
step = 0.05, 0.01
times = 2000
bins = 30
h2h = Hist2Hist4CMD('parsec', 'gaiaEDR3', bins)
h2p = Hist2Point4CMD('parsec', 'gaiaEDR3', bins)

lnlike_rand = pd.DataFrame(index=range(2000), columns=['t1_h2h', 't1_h2p', 't2_h2h', 't2_h2p'])

lnlike_rand['t1_h2h'] = test_randomness(theta1, n_stars, step, times, h2h)
lnlike_rand['t1_h2p'] = test_randomness(theta1, n_stars, step, times, h2p)

lnlike_rand['t2_h2h'] = test_randomness(theta2, n_stars, step, times, h2h)
lnlike_rand['t2_h2p'] = test_randomness(theta2, n_stars, step, times, h2p)

lnlike_rand.to_csv('/home/shenyueyue/Projects/starcat/test_data/lnlike_rand.csv', index=False)

sample_obs, med_nobs = read_sample_obs(
    '/home/shenyueyue/Projects/starcat/test_data/Cantat_2020/melotte_22_edr3.csv',
    'gaiaEDR3'
)
# !NOTE Gmag range
sample_obs = sample_obs[sample_obs['Gmag'] < 10]

# parameter
fb, dm = 0.35, 5.55
n_stars = 10000

# instantiate methods
parsec = Parsec()
isoc = Isoc(parsec)
imf = IMF('chabrier03')
binmethod = BinSimple()
photerr = GaiaEDR3('parsec', med_nobs)
# NOTE! changeable, Hist2Hist or Hist2Point
likelihoodfunc = Hist2Point4CMD('parsec', 'gaiaEDR3', 30)
synstars = SynStars('parsec', 'gaiaEDR3', imf, n_stars, binmethod, photerr)

# from bulkload_isocs.py
logage_grid = (6.6, 10, 0.01)
mh_grid = (-0.9, 0.7, 0.01)
lnlikes = likelihood_test.lnlike_2p_distrib(
    fb, dm, isoc, likelihoodfunc, synstars, logage_grid, mh_grid, sample_obs
)

# astart, aend, astep = logage_grid
# mstart, mend, mstep = mh_grid
# step = (astep, mstep)
# abin = np.arange(astart, aend, astep)
# mbin = np.arange(mstart, mend, mstep)
# logage_mh = []
# for a in abin:
#     for m in mbin:
#         logage_mh.append([a, m])
# print(f"calculate in total : {len(logage_mh)} lnlike values")
#
# lnlikes = []
# for theta_2p in logage_mh:
#     try:
#         lnlike = lnlike_2p(theta_2p, fb, dm, step, isoc, likelihoodfunc, synstars, sample_obs)
#         lnlikes.append(lnlike)
#     except ValueError:
#         print(theta_2p, fb, dm)

# theta_2p = [7.89, 0.032]
# lnlike = lnlike_2p(theta_2p, fb, dm, step, isoc, likelihoodfunc, synstars, sample_obs)
#
# lnlikes = []
# for theta_2p in logage_mh:
#     lnlike = lnlike_2p(theta_2p, fb, dm, step, isoc, likelihoodfunc, synstars, sample_obs)
#     lnlikes.append(lnlike)
#     print(theta_2p, fb, dm)
#
# df = pd.DataFrame(logage_mh, columns=['logage', 'mh'])
# df['lnlike'] = lnlikes
#
# i = isoc.get_isoc('gaiaEDR3',logage=6.609999999999999, mh=-0.9, logage_step=0.01, mh_step=0.01)
