import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from starcat import (Isoc, Parsec, IMF,
                     BinMRD,
                     CSSTsim, SynStars,
                     lnlike, Hist2Point4CMD, SahaW, DeltaCMD)

# initialization
photsys = 'CSST'
model = 'parsec'
imf = 'salpeter55'
imf_inst = IMF(imf)
parsec_inst = Parsec()
binmethod = BinMRD()
photerr = CSSTsim(model)
isoc_inst = Isoc(parsec_inst)
synstars_inst = SynStars(model, photsys,
                         imf_inst, binmethod, photerr)
h2p_cmd_inst = Hist2Point4CMD(model, photsys, bin_method='blocks')
sahaw_inst = SahaW(model, photsys, bin_method='knuth')
deltacmd_inst = DeltaCMD(model, photsys, bin_method='knuth')

likelihood_inst = deltacmd_inst

logage, mh, dm, Av, fb, alpha = 8., 0.0, 18.5, 0.5, 0.5, 2.35
theta_args = fb, alpha
theta = logage, mh, dm, Av, fb, alpha
step = (0.05, 0.05)  # logage, mh
logage_step, mh_step = step
n_stars = 1500
n_syn = 50000

samples = synstars_inst(theta, n_stars, isoc_inst)
observation = samples

obs_isoc = isoc_inst.get_obsisoc(photsys, dm=dm, Av=Av, logage=logage, mh=mh, logage_step=logage_step, mh_step=mh_step)

# %%
from scipy.interpolate import interp1d


def delta_color(sample, obs_isoc):
    isoc_c = obs_isoc['g'] - obs_isoc['i']
    isoc_m = obs_isoc['i']
    c = sample['g'] - sample['i']
    m = sample['i']
    isoc_line = interp1d(x=isoc_m, y=isoc_c, fill_value='extrapolate')
    temp_c = isoc_line(x=m)
    delta_c = c - temp_c
    sample['delta_c'] = delta_c
    return sample


observation = delta_color(observation, obs_isoc)

# %%
sample_dcmd = synstars_inst.delta_color_samples(theta, n_syn, isoc_inst, logage_step=logage_step, mh_step=mh_step)
Dcmd = lnlike(theta_args, step,
              isoc_inst, likelihood_inst, synstars_inst, n_syn,
              observation, 'LG', 5,
              logage=8., mh=0., dm=18.5, Av=0.5)

fig, ax = plt.subplots(figsize=(4, 5))
bin = observation['mass_sec'].notna()
c = observation['g'] - observation['i']
m = observation['i']

ax.scatter(c[bin], m[bin], color='#8E8BFE', s=5, alpha=0.6, label='binary')
ax.scatter(c[~bin], m[~bin], color='#E88482', s=5, alpha=0.6, label='single')
ax.plot(obs_isoc['g'] - obs_isoc['i'], obs_isoc['i'], color='r')
ax.set_ylim(max(observation['i']) + 0.2, min(observation['i']) - 0.5)
ax.set_xlim(min(observation['g'] - observation['i']) - 0.2, max(observation['g'] - observation['i']) + 0.2)
ax.legend()
ax.set_title(f'logage={logage}, [M/H]={mh}, DM={dm}, \n'
             f'Av={Av}, fb={fb}, alpha={alpha}', fontsize=12)
ax.set_xlabel('g - i')
ax.set_ylabel('i')

fig.show()

fig, ax = plt.subplots(figsize=(4, 5))
bin = sample_dcmd['mass_sec'].notna()
dc_syn = sample_dcmd['delta_c']
m_syn = sample_dcmd['i']
dc_obs = observation['delta_c']
m_obs = observation['i']

ax.scatter(dc_syn[bin], m_syn[bin], color='#8E8BFE', s=5, alpha=0.6, label='binary')
ax.scatter(dc_syn[~bin], m_syn[~bin], color='#E88482', s=5, alpha=0.6, label='single')
ax.scatter(dc_obs, m_obs, color='grey', s=5, alpha=0.6)
# ax.plot(obs_isoc['g']-obs_isoc['i'], obs_isoc['i'], color='r')
# ax.set_ylim(max(observation['i']) + 0.2, min(observation['i']) - 0.5)
# ax.set_xlim(min(observation['g'] - observation['i']) - 0.2, max(observation['g'] - observation['i']) + 0.2)
ax.invert_yaxis()
ax.legend(frameon=True)
ax.set_title(f'dCMD_H2P={Dcmd:.2f}', fontsize=12)
ax.set_xlabel('g - i')
ax.set_ylabel('i')

fig.show()
# %%
joblib.dump(observation,
            '/Users/sara/PycharmProjects/starcat/script/grid_likelihood/cl_dcmd_a8_m0_d18.5_a0.5_f0.5_a2.35.joblib')

fb_grid = np.arange(0.2, 1., 0.05)
alpha_grid = np.arange(1.6, 3, 0.05)

aa, bb = np.indices((len(fb_grid), len(alpha_grid)))
aa = aa.ravel()
bb = bb.ravel()

fb_vals = fb_grid[aa]
alpha_vals = alpha_grid[bb]

joint_like = np.zeros((len(fb_grid), len(alpha_grid)))


def calculate_likelihood(f, a):
    theta_args = f, a
    return lnlike(theta_args, step,
                  isoc_inst, likelihood_inst, synstars_inst, n_syn,
                  observation, 'LG', 5,
                  logage=8., mh=0., dm=18.5, Av=0.5)


# try:
#     with joblib_progress("calculating likelihood...", total=len(aa)):
#         results = zip(*Parallel(n_jobs=n_jobs)(
#             delayed(calculate_likelihood)(x, y) for x, y in zip(fb_vals, alpha_vals)
#
#         ))
# except:
#     joint_like[aa, bb] = results
#     joblib.dump(joint_like,
#                 '/Users/sara/PycharmProjects/starcat/script/grid_likelihood/cl_a8_m0_d18.5_a0.5_f0.5_a2.35.joblib')
#     text_content = (f'parallel stopped on the {len(results)} lnlike for 2P_likelihood\n')
#     with open('/Users/sara/PycharmProjects/starcat/script/grid_likelihood/' + 'output.txt', 'w') as file:
#         file.write(text_content)
#     print("details saved to output.txt")
# else:
#     joint_like[aa, bb] = results
#     joblib.dump(joint_like,
#                 '/Users/sara/PycharmProjects/starcat/script/grid_likelihood/cl_a8_m0_d18.5_a0.5_f0.5_a2.35.joblib')
results = pd.DataFrame(columns=['fb', 'alpha', 'log_like'])
log_like = []
for i, (x, y) in enumerate(zip(fb_vals, alpha_vals)):
    log_like.append(calculate_likelihood(x, y))
    if i % 10 == 0:
        print(i)

results['fb'] = fb_vals
results['alpha'] = alpha_vals
results['log_like'] = log_like
results['exp(log_like)'] = np.exp(log_like)
results.to_csv('/Users/sara/PycharmProjects/starcat/script/grid_likelihood/dcmd_a8_m0_d18.5_a0.5_f0.5_a2.35.csv',
               index=False)
# joint_like[aa, bb] = results
# joblib.dump(joint_like,
#             '/Users/sara/PycharmProjects/starcat/script/grid_likelihood/like_a8_m0_d18.5_a0.5_f0.5_a2.35.joblib')
#
#
# df = pd.DataFrame(joint_like)
# df.to_csv('/Users/sara/PycharmProjects/starcat/script/grid_likelihood/like_a8_m0_d18.5_a0.5_f0.5_a2.35.csv')
#

marginal_fb = results.groupby('fb')['exp(log_like)'].sum()

# 在 'alpha' 上的边缘分布
marginal_alpha = results.groupby('alpha')['exp(log_like)'].sum()

# 打印结果
print("在 'fb' 上的边缘分布:")
print(marginal_fb)

print("\n在 'alpha' 上的边缘分布:")
print(marginal_alpha)

fig, ax = plt.subplots()
ax.plot(list(fb_grid), list(marginal_fb))
ax.set_xlabel('fb')
ax.set_ylim(0, max(marginal_fb) * 1.005)
ax.axvline(x=0.5, c='r')
ax.set_ylabel('marginal exp(log_like)')
fig.show()

fig, ax = plt.subplots()
ax.plot(list(alpha_grid), list(marginal_alpha))
ax.set_xlabel('alpha')
ax.set_ylim(0, max(marginal_alpha) * 1.005)
ax.axvline(x=2.35, c='r')
ax.set_ylabel('marginal exp(log_like)')
fig.show()
