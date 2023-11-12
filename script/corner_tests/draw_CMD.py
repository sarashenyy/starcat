import joblib

dir = '/home/shenyueyue/Projects/starcat/data/corner/demo_age7/'
sample_path = 'sample_val.joblib'

logage = 7.0
mh = 0.0
sample = joblib.load(dir + sample_path)
# %%
import matplotlib.pyplot as plt
from matplotlib import gridspec
from starcat import config

model = 'parsec'
photsys = 'CSST'
source = config.config[model][photsys]
mag = source['mag']  # list
color = source['color']  # list

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 3)  # (nrows,ncols)
gs.update(wspace=0.3, hspace=0.2)

for i in range(len(mag)):
    single = sample[sample['q'].isna()]
    double = sample[~sample['q'].isna()]

    s_y = single[mag[i]]
    s_x = single[color[i][0]] - single[color[i][1]]
    d_y = double[mag[i]]
    d_x = double[color[i][0]] - double[color[i][1]]

    ax = plt.subplot(gs[int(i / 3), i % 3])  # facecolor='none'
    ax.scatter(d_x, d_y, s=5, color='c', alpha=0.3, label='double', marker='o', edgecolor='c')
    ax.scatter(s_x, s_y, s=2, color='m', alpha=0.9, label='single', marker='.')
    ax.invert_yaxis()

    ax.set_ylabel(mag[i], fontsize=16)
    ax.set_xlabel(f'{color[i][0]} - {color[i][1]}', fontsize=16)
    if i == 0:
        ax.legend()
    if i == 1:
        ax.set_title('$log_{10}{\\tau}$' + f'={logage}, [M/H]={mh}', fontsize=16)
plt.show()
