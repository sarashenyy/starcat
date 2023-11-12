import joblib
import numpy as np

from starcat import (Isoc, Parsec, IMF, BinMRD, CSSTsim,
                     SynStars,
                     lnlike_5p, Hist2Hist4CMD, Hist2Point4CMD)

dir = '/home/shenyueyue/Projects/starcat/data/corner/demo_age8/'
lnlikes_path = 'h2p_cmd(6).joblib'
sample_val_path = 'sample_val.joblib'

joint_lnlike = joblib.load(dir + lnlikes_path)
sample_val = joblib.load(dir + sample_val_path)

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
binmethod = BinMRD()
photerr = CSSTsim(model)
# ?init SynStars
synstars_inst = SynStars(model, photsys, imf_inst, binmethod, photerr)
# ?init LikelihoodFunc
h2h_cmd_inst = Hist2Hist4CMD(model, photsys, bins)
h2p_cmd_inst = Hist2Point4CMD(model, photsys, bins)

# %%
logage_val = 8.
mh_val = 0.
dist_val = 780.
Av_val = 0.5
fb_val = 0.5
n_val = 1500
theta_val = logage_val, mh_val, dist_val, Av_val, fb_val

logage_step = 0.1  # 0.1
mh_step = 0.1
step = (logage_step, mh_step)

logage = np.arange(7.5, 8.5, logage_step)
mh = np.arange(-0.9, 0.7, mh_step)
dist = np.arange(750, 850, 10)  # 10
Av = np.arange(0., 1., 0.1)  # 0.1
fb = np.arange(0.2, 1., 0.1)  # 0.1
times = 2
# times = 2
# print(len(logage), len(mh), len(dist), len(Av), len(fb))

ii, jj, aa, bb, cc, tt = np.indices((len(logage), len(mh), len(dist), len(Av), len(fb), times))
ii = ii.ravel()
jj = jj.ravel()
aa = aa.ravel()
bb = bb.ravel()
cc = cc.ravel()
tt = tt.ravel()

error_id = np.where(joint_lnlike > 0)

theta_list = []
lnlike_list = []
for i in range(len(error_id[0])):  # number of error_lnlikes
    age_aux = logage[error_id[0][i]]
    mh_aux = mh[error_id[1][i]]
    dist_aux = dist[error_id[2][i]]
    Av_aux = Av[error_id[3][i]]
    fb_aux = fb[error_id[4][i]]
    theta_aux = (age_aux, mh_aux, dist_aux, Av_aux, fb_aux)

    lnlike_aux = lnlike_5p(theta_aux, step, isoc_inst, h2p_cmd_inst, synstars_inst, n_stars, sample_val)
    theta_list.append(theta_aux)
    lnlike_list.append(lnlike_aux)

joint_lnlike[error_id] = lnlike_list

# %%

# ? when running on python console
from script.corner_tests.draw_corner import draw_corner

# ? when running on terminal
# from draw_corner import draw_corner
truth = list(theta_val)
parameters = [logage, mh, dist, Av, fb]

label = ['$log_{10}{\\tau}$', '[M/H]', '$d $(kpc)', '$A_{v}$', '$f_{b}$']
info_h2h_cmd = [photsys, 'kroupa01', 'BinMRD(uniform distribution)', 'Hist2Hist4CMD(6)', n_stars]
info_h2p_cmd = [photsys, 'kroupa01', 'BinMRD(uniform distribution)', 'Hist2Point4CMD(6)', n_stars]

draw_corner(truth=truth, parameters=parameters, ln_joint_distribution=joint_lnlike,
            label=label, info=info_h2p_cmd, savefig_path=dir + 'h2p_cmd(6).png')
