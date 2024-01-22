import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from starcat import (Isoc, Parsec, IMF,
                     BinMRD,
                     CSSTsim, SynStars,
                     lnlike, Hist2Point4CMD, SahaW, EnergyDistance, GaussianKDE)

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
h2p_cmd_inst = Hist2Point4CMD(model, photsys, bin_method='knuth')
sahaw_inst = SahaW(model, photsys, bin_method='knuth')
energy_inst = EnergyDistance(model, photsys, bin_method='knuth')
kde_inst = GaussianKDE(model, photsys)

likelihood_inst = h2p_cmd_inst

logage, mh, dm, Av, fb, alpha = 8., 0.0, 18.5, 0.5, 0.5, 2.35
theta_args = fb, alpha
theta = logage, mh, dm, Av, fb, alpha
step = (0.05, 0.05)  # logage, mh
n_stars = 1000
n_syn = 50000

samples = synstars_inst(theta, n_stars, isoc_inst)
observation = samples
fig, ax = plt.subplots(figsize=(4, 5))
bin = observation['mass_sec'].notna()
c = observation['g'] - observation['i']
m = observation['i']

ax.scatter(c[bin], m[bin], color='#8E8BFE', s=5, alpha=0.6, label='binary')
ax.scatter(c[~bin], m[~bin], color='#E88482', s=5, alpha=0.6, label='single')
ax.set_ylim(max(observation['i']) + 0.2, min(observation['i']) - 0.5)
ax.set_xlim(min(observation['g'] - observation['i']) - 0.2, max(observation['g'] - observation['i']) + 0.2)
ax.legend()
ax.set_title(f'logage={logage}, [M/H]={mh}, DM={dm}, \n'
             f'Av={Av}, fb={fb}, alpha={alpha}', fontsize=12)
ax.set_xlabel('g - i')
ax.set_ylabel('i')

fig.show()
joblib.dump(observation,
            '/Users/sara/PycharmProjects/starcat/script/grid_likelihood/cl_a8_m0_d18.5_a0.5_f0.5_a2.35.joblib')

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
                  observation, 'LG', 10,
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
    one = calculate_likelihood(x, y)
    log_like.append(one)
    print(i, one)
    # if i % 10 == 0:
    #     print(i)

results['fb'] = fb_vals
results['alpha'] = alpha_vals
results['log_like'] = log_like
results['exp(log_like)'] = np.exp(log_like)
# 转换为概率
max_l = max(results['log_like'])
p_ij = np.exp(results['log_like'] - max_l)
p_ij = p_ij / p_ij.sum()
results['p'] = p_ij

results.to_csv('/Users/sara/PycharmProjects/starcat/script/grid_likelihood/like1000_rgb_m0_d18.5_a0.5_f0.5_a2.35.csv',
               index=False)
# joint_like[aa, bb] = results
# joblib.dump(joint_like,
#             '/Users/sara/PycharmProjects/starcat/script/grid_likelihood/like_a8_m0_d18.5_a0.5_f0.5_a2.35.joblib')
#
#
# df = pd.DataFrame(joint_like)
# df.to_csv('/Users/sara/PycharmProjects/starcat/script/grid_likelihood/like_a8_m0_d18.5_a0.5_f0.5_a2.35.csv')
#

marginal_fb = results.groupby('fb')['p'].sum()

# 在 'alpha' 上的边缘分布
marginal_alpha = results.groupby('alpha')['p'].sum()

# 打印结果
print("在 'fb' 上的边缘分布:")
print(marginal_fb)

print("\n在 'alpha' 上的边缘分布:")
print(marginal_alpha)

fig, ax = plt.subplots()
ax.plot(list(fb_grid), list(marginal_fb))
ax.set_xlabel('$f_b$')
# ax.set_ylim(0, max(marginal_fb)*1.005)
ax.axvline(x=0.5, c='r')
ax.set_ylabel('marginal p')
fig.show()

fig, ax = plt.subplots()
ax.plot(list(alpha_grid), list(marginal_alpha))
ax.set_xlabel('$alpha$')
# ax.set_ylim(0, max(marginal_alpha)*1.005)
ax.axvline(x=2.35, c='r')
ax.set_ylabel('marginal p')
fig.show()

# %%
from matplotlib import gridspec

plt.style.use('/Users/sara/PycharmProjects/starcat/data/mystyle.mplstyle')

x = results['fb']
y = results['alpha']
z = results['p']
z = z.to_numpy().reshape(len(fb_grid), len(alpha_grid))
# 边缘分布
marginal_fb = results.groupby('fb')['p'].sum()
marginal_alpha = results.groupby('alpha')['p'].sum()
xlabel = '$f_b$'
ylabel = '$\\alpha$'

fig = plt.figure(figsize=(7, 6))
gs = gridspec.GridSpec(4, 4)
gs.update(hspace=0.08, wspace=0.08)

# hist of x (pdf)
ax_x = plt.subplot(gs[0, :-1])  # , sharex=ax_main
ax_x.plot(list(fb_grid), list(marginal_fb))
ax_x.axvline(x=0.5, c='r')
# ax_x.xaxis.set_visible(False)
ax_x.xaxis.tick_top()
ax_x.xaxis.set_label_position('top')
ax_x.set_ylabel('p(fb)')
left, right = ax_x.get_xlim()

# hist of y (pdf)
ax_y = plt.subplot(gs[1:, -1])  # , sharey=ax_main
ax_y.plot(list(marginal_alpha), list(alpha_grid))
ax_y.axhline(y=2.35, c='r')
ax_y.set_xlabel('p(alpha)')
# ax_y.yaxis.set_visible(False)
ax_y.yaxis.tick_right()
ax_y.yaxis.set_label_position('right')
bottom, top = ax_y.get_ylim()

# main
ax_main = plt.subplot(gs[1:, :-1])
ax_main.imshow(z.T, origin='lower', extent=(left, right, bottom, top),
               cmap='jet')  # ax.imshow(pp), pp.shape=(rows_Y, cols_X)
ax_main.set_xlabel(xlabel)
ax_main.set_ylabel(ylabel)

fig.show()
