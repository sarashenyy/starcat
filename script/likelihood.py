import pandas as pd

from starcat import (MagError, config,
                     GaiaEDR3, Parsec,
                     BinSimple,
                     Hist2Hist, Isoc, IMF, SynStars)
from starcat.test import likelihood_test


def read_sampel_obs():
    usecols = ['Gmag', 'BPmag', 'RPmag', 'phot_g_n_obs', 'phot_bp_n_obs', 'phot_rp_n_obs']
    sample = pd.read_csv(
        '/home/shenyueyue/Projects/starcat/test_data/melotte_22_dr2.csv',
        usecols=usecols
    )
    sample = sample.dropna().reset_index(drop=True)
    sample_temp = sample[['phot_g_n_obs', 'phot_bp_n_obs', 'phot_rp_n_obs']]
    nobs = MagError.extract_med_nobs(sample_temp, usecols[3:6])
    sample = sample[['Gmag', 'BPmag', 'RPmag']]
    sample.columns = config.config['parsec']['gaiaEDR3']['bands']
    return sample, nobs


sample_obs, med_nobs = read_sampel_obs()

# parameter
fb, dm = 0.35, 5.55
n_stars = 10000

# instantiate methods
parsec = Parsec()
isoc = Isoc(parsec)
imf = IMF('kroupa01')
binmethod = BinSimple()
photerr = GaiaEDR3('parsec', med_nobs)
# NOTE! changeable, Hist2Hist or Hist2Point
likelihoodfunc = Hist2Hist('parsec', 'gaiaDR2', 50)
synstars = SynStars('parsec', 'gaiaDR2', imf, n_stars, binmethod, photerr)

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
