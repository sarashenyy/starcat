import numpy as np
from matplotlib import pyplot as plt

from script.widgets import read_sample_obs
from starcat import (Parsec, Isoc, IMF, BinMS, BinSimple,
                     GaiaEDR3, Hist2Point, SynStars, cmd
                     )

sample_obs2, med_nobs2 = read_sample_obs('melotte_22_dr2', 'gaiaDR2')
sample_obs3, med_nobs3 = read_sample_obs('melotte_22_edr3', 'gaiaEDR3')

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

# hist2d的残差
model = 'parsec'
photsys = 'gaiaEDR3'
sample_obs = sample_obs3
sample_syn = sample_syn3

c_obs, m_obs = cmd.CMD.extract_cmd(sample_obs, model, photsys)
c_syn, m_syn = cmd.CMD.extract_cmd(sample_syn, model, photsys)
obs = plt.hist2d(c_obs, m_obs, cmap='Blues')
syn = plt.hist2d(c_syn, m_syn, cmap='Blues')

# 计算obs和syn的差异
residuals = obs[0] / np.sum(obs[0]) - syn[0] / np.sum(syn[0])

# 绘制残差图
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(residuals, cmap='bwr', origin='lower')
cbar = fig.colorbar(im, ax=ax, label='Residuals')
ax.set_xlabel('Color')
ax.set_ylabel('Magnitude')
ax.set_title('Histogram2D Residuals')
ax.invert_yaxis()

fig.show()
