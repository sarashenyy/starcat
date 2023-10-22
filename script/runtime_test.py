# %%
import timeit

from starcat import (config,
                     Isoc, Parsec, IMF, BinMRD, CSSTsim,
                     SynStars,
                     lnlike_5p, Hist2Hist4CMD, Hist2Point4CMD, Hist2Hist4Bands)

# !define instance from starcat
n_stars = 5000
bins = 50
step_lnlike = 0.5
photsys = 'CSST'
model = 'parsec'
# ?init isochrone
parsec_inst = Parsec()
isoc_inst = Isoc(parsec_inst)
# ?init IMF
imf_inst = IMF('kroupa01')
# ?init Binmethod & photometric error system
binmethod = BinMRD()
photerr = CSSTsim(model)
# ?init SynStars
synstars_inst = SynStars(model, photsys, imf_inst, n_stars, binmethod, photerr)
# ?init LikelihoodFunc
h2h_cmd_inst = Hist2Hist4CMD(model, photsys, bins)
h2p_cmd_inst = Hist2Point4CMD(model, photsys, bins)
# h2h_bds_inst = Hist2Hist4Bands(model, photsys, step = step_lnlike)
h2h_bds_inst = Hist2Hist4Bands(model, photsys)

# !create synthetic cluster for validation
logage_val = 7.
mh_val = 0.
dist_val = 780.
Av_val = 0.5
fb_val = 0.5
n_val = 1300
theta_val = logage_val, mh_val, dist_val, Av_val, fb_val

source = config.config[model][photsys]
bands = source['bands']
mag = source['mag']
color = source['color']
mag_max = source['mag_max']

# ! synstars val
synstars_val = SynStars(model, photsys, imf_inst, n_val, binmethod, photerr)
isoc_ori = isoc_inst.get_isoc(photsys, logage=logage_val, mh=mh_val)
# # ?synthetic isochrone (distance and Av added)
# isoc_val = synstars_val.get_observe_isoc(isoc_ori, dist_val, Av_val)
#
# # ?synthetic cluster sample (without error added)
# sample_val_noerr = synstars_val.sample_stars(isoc_val, fb_val)

# ?synthetic cluster sample (with phot error)
sample_val = synstars_val(theta_val, isoc_ori)
sample_val = sample_val[sample_val[mag] <= 25.5]

# %%
# * synstars_val
%timeit
synstars_val(theta_val, isoc_ori)

# !define instance from starcat
theta_5p = (7.5, 0., 760., 0.3, 0.5)
logage_step = 0.1
mh_step = 0.05
step = (logage_step, mh_step)

times = 50
# * lnlike_5p
%timeit - n
10
lnlike_5p(theta_5p, step, isoc_inst, h2h_cmd_inst, synstars_inst, sample_val, times)
%timeit - n
10
lnlike_5p(theta_5p, step, isoc_inst, h2p_cmd_inst, synstars_inst, sample_val, times)
%timeit - n
10
lnlike_5p(theta_5p, step, isoc_inst, h2h_bds_inst, synstars_inst, sample_val, times)
