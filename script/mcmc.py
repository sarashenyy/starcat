import logging
from multiprocessing import Pool

import emcee
import numpy as np
import pandas as pd

from starcat import (MagError, config,
                     GaiaEDR3, Parsec, BinMS, Hist2Hist,
                     Isoc, IMF, SynStars, lnlike_2p)
from starcat.widgets import log_time

# cofiguring log output to logfile
# module_dir = os.path.dirname(__file__)
# filename = os.path.basename(__file__)
# filename_without_extension = os.path.splitext(filename)[0]
# log_file = os.path.join(module_dir, '..', 'logs', f'{filename_without_extension}.log')
log_file = '/home/shenyueyue/Projects/starcat/log/mcmc.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    filemode='w')


def read_sampe_obs():
    usecols = ['Gmag', 'BPmag', 'RPmag', 'phot_g_n_obs', 'phot_bp_n_obs', 'phot_rp_n_obs']
    sample = pd.read_csv(
        '/home/shenyueyue/Projects/starcat/test_data/melotte_22_edr3.csv',
        usecols=usecols
    )
    sample = sample.dropna().reset_index(drop=True)
    sample_temp = sample[['phot_g_n_obs', 'phot_bp_n_obs', 'phot_rp_n_obs']]
    nobs = MagError.extract_med_nobs(sample_temp, usecols[3:6])
    sample = sample[['Gmag', 'BPmag', 'RPmag']]
    sample.columns = config.config['parsec']['gaiaEDR3']['bands']
    return sample, nobs


sample_obs, med_nobs = read_sampe_obs()

# parameter
theta_2p = (7.89, 0.032)
fb, dm = 0.35, 5.55
step = (0.05, 0.2)
n_stars = 10000

# initial methos
parsec = Parsec()
isoc = Isoc(parsec)
imf = IMF('kroupa01')
binmethod = BinMS()
photerr = GaiaEDR3('parsec', med_nobs)
likelihoodfunc = Hist2Hist('parsec', 'gaiaEDR3', 30, 100)
synstars = SynStars('parsec', 'gaiaEDR3', imf, n_stars, binmethod, photerr)


@log_time
def main():
    lnlike_2p(theta_2p, fb, dm, step, isoc, likelihoodfunc, synstars, sample_obs)


# MCMC
ndim = 2
# set up MCMC sampler
nwalkers = 100
scale = np.array([0.1, 0.1])
p0 = np.round((theta_2p + scale * np.random.randn(nwalkers, ndim)), 2)
# parallelzation
with Pool() as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnlike_2p,
        args=(fb, dm, step, isoc, likelihoodfunc, synstars, sample_obs)
    )
    nburn = 200
    pos, _, _, = sampler.run_mcmc(p0, nburn, progress=True)
