import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import ListedColormap

plt.style.use('/Users/sara/PycharmProjects/starcat/data/mystyle.mplstyle')
cmap = ListedColormap(sns.color_palette("RdBu_r", n_colors=256))

logage_grid = np.arange(6.7, 9.0, 0.5)
mh_grid = np.arange(-1.0, 0.4, 0.2)
dm_grid = np.arange(4.0, 8.0, 0.5)
Av_grid = np.arange(0.0, 2.0, 0.25)
fb_grid = np.arange(0.0, 1.0, 0.2)
alpha_grid = np.arange(0.5, 3.0, 0.5)

parameters = [logage_grid, mh_grid, dm_grid, Av_grid, fb_grid, alpha_grid]
truth = [8.0, 0.0, 5.5, 0.35, 0.2, 2.3]
label = ['log(age)', '[M/H]', 'DM', '$A_{v}$', '$f_{b}$', '$\\alpha$']

# shift joint_lnlike to avoid overflow in np.exp
# joint_lnlike_noinf = joint_lnlike[joint_lnlike != -np.inf]
# print(f'joint_lnlike: [-inf, {np.min(joint_lnlike_noinf)}, {np.max(joint_lnlike_noinf)}]')
# min_lnlike = np.min(joint_lnlike[joint_lnlike!=-np.inf])
# print('min_lnlike:', min_lnlike)
# joint_lnlike_shifted = (joint_lnlike - min_lnlike) + 1
# print(f'joint_lnlike_shifted: {np.min(joint_lnlike_shifted)} {np.max(joint_lnlike_shifted)}')
# joint_like = np.exp(joint_lnlike_shifted)
# print(f'joint_like: {np.min(joint_like)} {np.max(joint_like)}')

# joint_like = np.exp(joint_lnlike)
joint_lnlike_noinf = joint_lnlike[joint_lnlike != -np.inf]
print(f'joint_lnlike: [-inf, {np.min(joint_lnlike_noinf)}, {np.max(joint_lnlike_noinf)}]')
max_lnlike = np.max(joint_lnlike)
print('min_lnlike:', max_lnlike)
joint_lnlike_shifted = (joint_lnlike - max_lnlike) + 10
print(
    f'joint_lnlike_shifted: [-inf, {np.min(joint_lnlike_shifted[joint_lnlike_shifted != -np.inf])}, {np.max(joint_lnlike_shifted)}]')
joint_like = np.exp(joint_lnlike_shifted)
print(f'joint_like: [0.0, {np.min(joint_like[joint_lnlike_shifted != -np.inf])} {np.max(joint_like)}]')

num_subplot = len(joint_lnlike.shape)

fig = plt.figure(figsize=(15, 17))  # (Width, height)
gs = gridspec.GridSpec(num_subplot, num_subplot)  # (nrows,ncols)
# gs.update(wspace=0.06, hspace=0.06)

rows_to_draw = np.arange(num_subplot)
for col in range(num_subplot):

    # ! calculate marginal distribution of single parameter
    # get x axis data
    data_x = parameters[col]
    # get y axis data (marginal distribution of p)
    marginal_p = np.sum(joint_like, axis=tuple(set(np.arange(num_subplot)) - {col}))
    # normalize
    marginal_p = marginal_p / np.sum(marginal_p)
    # ? draw diagonal plots with marginal distribution of single parameter
    ax_diagonal = plt.subplot(gs[col, col])
    ax_diagonal.plot(data_x, marginal_p, c='k')
    ax_diagonal.scatter(data_x, marginal_p, c='grey')
    ax_diagonal.axvline(truth[col], c='r', linestyle='--')

    # 将 y 轴刻度值放置在右侧
    ax_diagonal.yaxis.tick_right()
    ax_diagonal.xaxis.set_label_position('top')
    ax_diagonal.set_xlabel(label[col], fontsize=16)
    # to control the x extent in non-diagonal subplots
    left, right = ax_diagonal.get_xlim()

    # ! calculate marginal distribution of two parameters
    if col < num_subplot - 1:
        rows_to_draw = rows_to_draw[1:]
        for row in rows_to_draw:
            marginal_pp = np.sum(joint_like, axis=tuple(set(np.arange(num_subplot)) - {col, row}))
            ln_marginal_pp = np.log(marginal_pp)
            ln_marginal_pp = np.transpose(ln_marginal_pp)
            # print(f'row,col={row, col}, axis={tuple(set(np.arange(num_subplot)) - {col, row})}, '
            #       f'marginal_pp={marginal_pp.shape}')
            data_y = parameters[row]
            x_grid, y_grid = np.meshgrid(data_x, data_y)

            # ? draw non diagonal plots with marginal distribution of two parameters
            ax = plt.subplot(gs[row, col])
            ax.scatter(x_grid, y_grid, color='gray', alpha=0.5, s=2)
            # to control the y extent in non-diagonal subplots
            bottom, top = ax.get_ylim()

            # * NOTE! ax.contour(X,Y,Z)
            # * NOTE! X and Y must both be 2D with the same shape as Z (e.g. created via numpy.meshgrid)
            ax.contour(x_grid, y_grid, ln_marginal_pp, colors='black', linewidths=0.5, linestyles='-',
                       extent=(left, right, bottom, top),
                       origin='lower')

            # * NOTE! ax.imshow(pp), pp.shape=(rows_Y, cols_X)
            # * NOTE! ax.imshow(origin='lower')
            # *       control which point in pp represents the original point in figure(lower / upper)
            im = ax.imshow(ln_marginal_pp, cmap=cmap,
                           extent=(left, right, bottom, top),
                           origin='lower')
            cbar = plt.colorbar(im, ax=ax, location='top')
            cbar.ax.tick_params(labelsize=8)
            cbar.formatter.set_useOffset(False)
            cbar.update_ticks()

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
            # ax.tick_params(axis='x', direction='out')
            # ax.tick_params(axis='y', direction='out')

            if row == num_subplot - 1:
                ax.set_xlabel(label[col], fontsize=16)
                ax.tick_params(axis='x', labelsize=8)
            if col == 0:
                ax.set_ylabel(label[row], fontsize=16)
                ax.tick_params(axis='y', labelsize=8)

# fig.text(0.5, 0.7, f'Truth: {truth}\n\n'
#                    f'Photsys: {info[0]}\nIMF: {info[1]}\nBinary: {info[2]}\nLikelihood: {info[3]}\n'
#                    f'Synthetic star number: {info[4]}',
#          fontsize=16, ha='left')
plt.savefig('/Users/sara/PycharmProjects/starcat/script/grid_likelihood/synsample_h2hv1_binfixed.png',
            bbox_inches='tight')
# fig.show()
plt.close(fig)
