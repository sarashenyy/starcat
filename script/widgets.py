import pandas as pd

from starcat import (Parsec, Isoc, IMF, config,
                     BinSimple,
                     MagError,
                     SynStars
                     )
from starcat.photerr import GaiaDR3


def read_sample_obs(filename, photsys):
    usecols = ['Gmag', 'BPmag', 'RPmag', 'phot_g_n_obs', 'phot_bp_n_obs', 'phot_rp_n_obs']
    sample = pd.read_csv(
        f'{filename}',
        usecols=usecols
    )
    sample = sample.dropna().reset_index(drop=True)
    sample_temp = sample[['phot_g_n_obs', 'phot_bp_n_obs', 'phot_rp_n_obs']]
    nobs = MagError.extract_med_nobs(sample_temp, usecols[3:6])
    sample = sample[['Gmag', 'BPmag', 'RPmag']]
    sample.columns = config.config['parsec'][photsys]['bands']
    return sample, nobs


def get_sample_syn(filename, photsyn, n_stars):
    sample_obs3, med_nobs3 = read_sample_obs(filename, 'gaiaEDR3')
    theta = 7.89, 0.032, 0.35, 5.55
    step = 0.01, 0.01
    # !NOTE n_stars
    # instantiate methods
    parsec = Parsec()
    isoc = Isoc(parsec)
    imf = IMF('kroupa01')
    binmethod = BinSimple()

    photerr3 = GaiaDR3('parsec', med_nobs3)
    synstars3 = SynStars('parsec', photsyn, imf, n_stars, binmethod, photerr3)

    sample_syn = synstars3(theta, step, isoc)
    sample_syn = sample_syn[['Gmag', 'G_BPmag', 'G_RPmag']]
    return sample_syn
