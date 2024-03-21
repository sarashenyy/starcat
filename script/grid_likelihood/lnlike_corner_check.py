parameters
logage_grid, mh_grid, dm_grid, Av_grid, fb_grid, alpha_grid = parameters

logage_len = len(logage_grid)
mh_len = len(mh_grid)
dm_len = len(dm_grid)
Av_len = len(Av_grid)
fb_len = len(fb_grid)
alpha_len = len(alpha_grid)

aa, bb, cc, dd, ee, ff = np.indices((logage_len, mh_len, dm_len, Av_len, fb_len, alpha_len))
aa = aa.ravel()
bb = bb.ravel()
cc = cc.ravel()
dd = dd.ravel()
ee = ee.ravel()
ff = ff.ravel()

logage_vals = logage_grid[aa]
mh_vals = mh_grid[bb]
dm_vals = dm_grid[cc]
Av_vals = Av_grid[dd]
fb_vals = fb_grid[ee]
alpha_vals = alpha_grid[ff]

args_list = []
for i in range(len(logage_vals)):
    logage_val = logage_vals[i]
    mh_val = mh_vals[i]
    dm_val = dm_vals[i]
    Av_val = Av_vals[i]
    fb_val = fb_vals[i]
    alpha_val = alpha_vals[i]
    args_list.append((logage_val, mh_val, dm_val, Av_val, fb_val, alpha_val))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

joint_lnlike_df = pd.DataFrame([], columns=['logage', 'mh', 'dm', 'Av', 'fb', 'alpha', 'joint_lnlike'])
joint_lnlike_df['logage'] = logage_vals
joint_lnlike_df['mh'] = mh_vals
joint_lnlike_df['dm'] = dm_vals
joint_lnlike_df['Av'] = Av_vals
joint_lnlike_df['fb'] = fb_vals
joint_lnlike_df['alpha'] = alpha_vals
joint_lnlike_df['joint_lnlike'] = results
shift_value = np.min(results)
results_shifted = (results - shift_value) + 1
joint_lnlike_df['joint_like'] = np.exp(results_shifted)

# %%
# 检验二维边缘分布
row, col = 'alpha', 'fb'
aux = joint_lnlike_df.groupby([row, col])['joint_like'].sum()
# 创建一个新的 DataFrame 包含唯一的 ('logage', 'mh') 组合以及计算出的总和
aux_df = pd.DataFrame({
    row: aux.index.get_level_values(row),
    col: aux.index.get_level_values(col),
    'marginal_like': aux.values,
    'marginal_lnlike': np.log(aux.values)
})

fig, ax = plt.subplots(figsize=(6, 6.5))
scatter = ax.scatter(aux_df[col], aux_df[row], c=aux_df['marginal_lnlike'],
                     cmap=cmap, s=5000, marker='s')
cbar = plt.colorbar(scatter, location='top')
ax.set_xlabel(col)
ax.set_ylabel(row)
cbar.formatter.set_useOffset(False)
cbar.update_ticks()
fig.show()

# %%
# 检验一维边缘分布
col = 'alpha'
aux = joint_lnlike_df.groupby([col])['joint_like'].sum()
aux_df = pd.DataFrame({
    col: aux.index.get_level_values(col),
    'marginal_like': aux.values,
    'marginal_like_norm': aux.values / np.sum(aux.values)
})

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(aux_df[col], aux_df['marginal_like_norm'], c='grey')
ax.plot(aux_df[col], aux_df['marginal_like_norm'], c='k')
ax.set_xlabel(col)
fig.show()
