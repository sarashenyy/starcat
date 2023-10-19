import time

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib import gridspec
from scipy.stats import multivariate_normal

start_time = time.time()

# * TEST corner.py WITH three-dimensional standard normal distribution
# 创建三维坐标网格
x = np.arange(-5, 5, 0.5)
y = np.arange(-5, 5, 0.5)
z = np.arange(-5, 5, 0.5)

ii, jj, aa = np.indices((len(x), len(y), len(z)))
ii = ii.ravel()
jj = jj.ravel()
aa = aa.ravel()

x_vals = x[ii]
y_vals = y[jj]
z_vals = z[aa]

# 创建均值向量
mean = np.array([0, 0, 0])
# 创建协方差矩阵（单位矩阵）
cov = np.eye(3)
# 创建多元正态分布对象
mvn = multivariate_normal(mean=mean, cov=cov)

# 初始化似然值的数组
likelihood_values = np.zeros((len(x), len(y), len(z)))

# 计算似然值并行化
n_jobs = -1
results = Parallel(n_jobs=n_jobs)(
    delayed(mvn.pdf)([xi, yi, zi]) for xi, yi, zi in zip(x_vals, y_vals, z_vals)
)

# 将结果写入 likelihood_values
likelihood_values[ii, jj, aa] = results

end_time = time.time()
run_time = end_time - start_time
print(f"运行时间: {run_time} 秒")

# %%
# * CHANGE SOME PROPERTIES FOR CORNER
truth = list(mean)
parameters = [x, y, z]
joint_distribution = likelihood_values
num_subplot = len(joint_distribution.shape)
label = ['x', 'y', 'z']

# ! NOTHING NEED TO REWRITE BELOW
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(num_subplot, num_subplot)  # (nrows,ncols)
gs.update(wspace=0.06, hspace=0.06)

rows_to_draw = np.arange(num_subplot)
for col in range(num_subplot):
    # ! calculate marginal distribution of single parameter
    # get x axis data
    data_x = parameters[col]
    # get y axis data (marginal distribution of p)
    marginal_p = np.sum(joint_distribution, axis=tuple(set(np.arange(num_subplot)) - {col}))
    # normalize
    marginal_p = marginal_p / np.sum(marginal_p)
    # ? draw diagonal plots with marginal distribution of single parameter
    ax_diagonal = plt.subplot(gs[col, col])
    ax_diagonal.plot(data_x, marginal_p, c='k')
    ax_diagonal.scatter(data_x, marginal_p, c='grey')
    ax_diagonal.axvline(truth[col], c='r', linestyle='--')

    # ax_diagonal.set_xticklabels([])
    # ax_diagonal.set_yticklabels([])
    # ax_diagonal.tick_params(axis='x', direction='in')
    # ax_diagonal.tick_params(axis='y', direction='in')
    #
    # if col == num_subplot - 1:
    #     ax_diagonal.set_xlabel(label[col])
    # if col == 0:
    #     ax_diagonal.set_ylabel(label[col])
    if col != num_subplot - 1:
        # ax_diagonal.get_xaxis().set_visible(False)
        ax_diagonal.set_xticklabels([])
    if col != 0:
        ax_diagonal.get_yaxis().set_visible(False)
        ax_diagonal.set_yticklabels([])
    ax_diagonal.tick_params(axis='x', direction='out')
    ax_diagonal.tick_params(axis='y', direction='out')

    ax_diagonal.set_xlabel(label[col])
    ax_diagonal.set_ylabel(label[col])

    # ! calculate marginal distribution of two parameters
    if col < num_subplot - 1:
        rows_to_draw = rows_to_draw[1:]
        for row in rows_to_draw:
            marginal_pp = np.sum(joint_distribution, axis=tuple(set(np.arange(num_subplot)) - {col, row}))
            marginal_pp = np.transpose(marginal_pp)
            # print(f'row,col={row, col}, axis={tuple(set(np.arange(num_subplot)) - {col, row})}, '
            #       f'marginal_pp={marginal_pp.shape}')
            data_y = parameters[row]
            x_grid, y_grid = np.meshgrid(data_x, data_y)
            # ? draw non diagonal plots with marginal distribution of two parameters
            ax = plt.subplot(gs[row, col])
            ax.scatter(x_grid, y_grid, color='gray', alpha=0.5, s=2)
            # ax.scatter(x_grid, y_grid, c=marginal_pp, cmap='viridis')
            # * NOTE! ax.imshow(pp), pp.shape=(rows_Y, cols_X)
            ax.imshow(marginal_pp, cmap='GnBu', extent=(data_x.min(), data_x.max(), data_y.min(), data_y.max()))
            # * NOTE! ax.contour(X,Y,Z)
            # * NOTE! X and Y must both be 2D with the same shape as Z (e.g. created via numpy.meshgrid)
            ax.contour(x_grid, y_grid, marginal_pp, colors='black', linewidths=0.5, linestyles='-',
                       extent=(data_x.min(), data_x.max(), data_y.min(), data_y.max()))
            ax.set_aspect('auto')
            ax.axvline(truth[col], c='r', linestyle='--')
            ax.axhline(truth[row], c='r', linestyle='--')
            ax.plot(truth[col], truth[row], 'sr')

            # ax.tick_params(axis='x', direction='in')
            # ax.tick_params(axis='y', direction='in')
            # if row == num_subplot - 1:
            #     ax.set_xlabel(label[col])
            # if col == 0:
            #     ax.set_ylabel(label[row])
            if row != num_subplot - 1:
                # ax.get_xaxis().set_visible(False)
                ax.set_xticklabels([])
            if col != 0:
                # ax.get_yaxis().set_visible(False)
                ax.set_yticklabels([])
            ax.tick_params(axis='x', direction='out')
            ax.tick_params(axis='y', direction='out')

            if row == num_subplot - 1:
                ax.set_xlabel(label[col])
            if col == 0:
                ax.set_ylabel(label[row])

plt.show()
