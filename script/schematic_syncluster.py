# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from starcat import (config,
                     Isoc, Parsec, IMF,
                     BinMRD, CSSTsim,
                     SynStars
                     )

# %%

logage = [7, 8, 9]
mh = 0.0
dm = 24.45
fb = 0.35
gq = 0
n_stars = 1500
photsyn = 'CSST'
model = 'parsec'
base_path = '/home/shenyueyue/Projects/starcat/test_data/CSSTsim/'

# %%

# ?init isochrone
parsec_inst = Parsec()
isoc_inst = Isoc(parsec_inst)

for i in range(len(logage)):
    isoc = isoc_inst.get_isoc(photsyn, logage=logage[i], mh=mh)
    isoc.to_csv(
        f'{base_path}isochrones/age{logage[i]:+.2f}_mh{mh:+.2f}.csv',
        index=False
    )

# %%

# ?draw isoc with different logage
source = config.config[model][photsyn]
bands = source['bands']
mag = source['mag']
color = source['color']
mag_max = source['mag_max']
line_styles = ['-', '-.', '--']
mass_to_plot = [1, 2, 2.3, 3, 5.3, 6, 17.1]

fig, ax = plt.subplots(figsize=(4, 6))

for i in range(len(logage)):
    isoc = pd.read_csv(
        f'{base_path}isochrones/age{logage[i]:+.2f}_mh{mh:+.2f}.csv'
    )
    c = isoc[color[0]] - isoc[color[1]]
    m = isoc[mag] + dm
    ax.plot(c, m, linestyle=line_styles[i], label=f'logage={logage[i]}')
    index = np.isin(isoc['Mass'], mass_to_plot)
    ax.scatter(c[index], m[index], c='k', s=10, zorder=5)

x_min, x_max = ax.get_xlim()
ax.axhline(mag_max, color='r', linestyle=':', label=f'{mag_max}(mag)')

ax.invert_yaxis()
ax.grid(True, linestyle='--')
ax.set_xlabel('g - i')
ax.set_ylabel('i')
ax.set_title('DM=24.45, distance=778kpc')
ax.legend()
fig.show()

# %%

# ?draw isoc with different phase
phase = source['phase']
color_list = ['grey', 'green', 'orange', 'red', 'blue', 'skyblue', 'pink', 'purple', 'grey', 'black']
line_styles = ['-', '-.', '--']

fig, ax = plt.subplots(figsize=(4, 6))

for i in range(len(logage)):
    isoc = pd.read_csv(
        f'{base_path}isochrones/age{logage[i]:+.2f}_mh{mh:+.2f}.csv'
    )
    c = isoc[color[0]] - isoc[color[1]]
    m = isoc[mag] + dm
    if i == 0:
        for j, element in enumerate(phase):
            index = isoc['phase'] == element
            ax.plot(c[index], m[index], linestyle=line_styles[i], color=color_list[j], label=element)
    else:
        for j, element in enumerate(phase):
            index = isoc['phase'] == element
            ax.plot(c[index], m[index], linestyle=line_styles[i], color=color_list[j])

x_min, x_max = ax.get_xlim()
ax.axhline(mag_max, color='r', linestyle=':', label=f'{mag_max}(mag)')

ax.invert_yaxis()
ax.grid(True, linestyle='--')
ax.set_xlabel('g - i')
ax.set_ylabel('i')
ax.set_title('DM=24.45, distance=778kpc')
ax.legend()
fig.show()

# %%

logage = [7, 8, 9]
mh = 0.0
dm = 24.45
fb = 0.35
gq = 0
n_stars = 1500
photsyn = 'CSST'
model = 'parsec'

source = config.config[model][photsyn]
bands = source['bands']
mag = source['mag']
color = source['color']
mag_max = source['mag_max']

# ?init isochrone
parsec_inst = Parsec()
isoc_inst = Isoc(parsec_inst)
# ?init IMF
imf_inst = IMF('kroupa01')
# ?init Binmethod & photometric error system
binmethod = BinMRD()
photerr = CSSTsim(model)
# ?init SynStars
synstars_inst = SynStars(model, photsyn, imf_inst, n_stars, binmethod, photerr)

# fig = plt.figure(figsize=(10, 12))
# gs = gridspec.GridSpec(3, 2)

for i in range(len(logage)):
    theta = logage[i], mh, fb, dm
    isoc = pd.read_csv(
        f'{base_path}isochrones/age{logage[i]:+.2f}_mh{mh:+.2f}.csv'
    )

    sample0 = synstars_inst.sample_stars(isoc, fb, dm)
    for _ in bands:
        sample0[_] += dm
    sample0.to_csv(
        f'{base_path}synclusters/age{logage[i]:+.2f}_mh{mh:+.2f}_sample0.csv',
        index=False
    )

    sample1 = synstars_inst(theta, isoc)
    sample1.to_csv(
        f'{base_path}synclusters/age{logage[i]:+.2f}_mh{mh:+.2f}_sample1.csv',
        index=False
    )
# ax = plt.subplot(gs[i, 0])
# ax.scatter()
