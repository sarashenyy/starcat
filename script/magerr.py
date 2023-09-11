import pandas as pd

from starcat import (Parsec, Isoc, IMF, config,
                     BinMS, BinSimple,
                     GaiaEDR3, MagError,
                     Hist2Point, SynStars
                     )


def read_sampel_obs(filename, photsys):
    usecols = ['Gmag', 'BPmag', 'RPmag', 'phot_g_n_obs', 'phot_bp_n_obs', 'phot_rp_n_obs']
    sample = pd.read_csv(
        f'/home/shenyueyue/Projects/starcat/test_data/{filename}.csv',
        usecols=usecols
    )
    sample = sample.dropna().reset_index(drop=True)
    sample_temp = sample[['phot_g_n_obs', 'phot_bp_n_obs', 'phot_rp_n_obs']]
    nobs = MagError.extract_med_nobs(sample_temp, usecols[3:6])
    sample = sample[['Gmag', 'BPmag', 'RPmag']]
    sample.columns = config.config['parsec'][photsys]['bands']
    return sample, nobs


sample_obs2, med_nobs2 = read_sampel_obs('melotte_22_dr2', 'gaiaDR2')
sample_obs3, med_nobs3 = read_sampel_obs('melotte_22_edr3', 'gaiaEDR3')

theta = 7.89, 0.032, 0.35, 5.55
step = 0.01, 0.01
n_stars = 1000
# instantiate methods
parsec = Parsec()
isoc = Isoc(parsec)
imf = IMF('kroupa01')
binmethod = BinSimple()
binmethod_ms = BinMS()

photerr2 = GaiaEDR3('parsec', med_nobs2)
likelihoodfunc2 = Hist2Point('parsec', 'gaiaDR2', 50)
synstars2 = SynStars('parsec', 'gaiaDR2', imf, n_stars, binmethod, photerr2)

photerr3 = GaiaEDR3('parsec', med_nobs3)
likelihoodfunc3 = Hist2Point('parsec', 'gaiaEDR3', 50)
synstars3 = SynStars('parsec', 'gaiaEDR3', imf, n_stars, binmethod, photerr3)

synstars3_ms = SynStars('parsec', 'gaiaEDR3', imf, n_stars, binmethod_ms, photerr3)

sample_syn2 = synstars2(theta, step, isoc)
sample_syn3 = synstars3(theta, step, isoc)
sample_syn3_ms = synstars3_ms(theta, step, isoc)
