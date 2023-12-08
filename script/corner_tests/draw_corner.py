import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

plt.style.use('/Users/sara/PycharmProjects/starcat/data/mystyle.mplstyle')


def draw_corner(truth, parameters, ln_joint_distribution, label, info, savefig_path):
    # * NOTE correction, make max(ln_joint_distribution)=0
    delta = np.max(ln_joint_distribution)
    ln_joint_distribution = ln_joint_distribution - delta

    joint_distribution = np.exp(ln_joint_distribution)
    num_subplot = len(joint_distribution.shape)

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

        ax_diagonal.set_xlabel(label[col], fontsize=16)
        ax_diagonal.set_ylabel(label[col], fontsize=16)
        # to control the x extent in non-diagonal subplots
        left, right = ax_diagonal.get_xlim()

        # ! calculate marginal distribution of two parameters
        if col < num_subplot - 1:
            rows_to_draw = rows_to_draw[1:]
            for row in rows_to_draw:
                # marginal_pp = np.sum(joint_distribution, axis=tuple(set(np.arange(num_subplot)) - {col, row}))
                # marginal_pp = np.transpose(marginal_pp)
                # ln_marginal_pp = np.sum(ln_joint_distribution, axis=tuple(set(np.arange(num_subplot)) - {col, row}))
                ln_marginal_pp = np.sum(joint_distribution, axis=tuple(set(np.arange(num_subplot)) - {col, row}))
                ln_marginal_pp = np.log(np.where(ln_marginal_pp <= 0, 1e-10, ln_marginal_pp))
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
                # ax.contour(x_grid, y_grid, marginal_pp, colors='black', linewidths=0.5, linestyles='-',
                #            extent=(data_x.min(), data_x.max(), data_y.min(), data_y.max()))
                # ax.contour(x_grid, y_grid, marginal_pp, colors='black', linewidths=0.5, linestyles='-',
                #            extent=(data_x.min(), data_x.max(), data_y.min(), data_y.max()), origin='lower')
                ax.contour(x_grid, y_grid, ln_marginal_pp, colors='black', linewidths=0.5, linestyles='-',
                           extent=(left, right, bottom, top),
                           origin='lower')

                # * NOTE! ax.imshow(pp), pp.shape=(rows_Y, cols_X)
                # * NOTE! ax.imshow(origin='lower')
                # *       control which point in pp represents the original point in figure(lower / upper)
                # ax.imshow(marginal_pp, cmap='GnBu', extent=(data_x.min(), data_x.max(), data_y.min(), data_y.max()))
                # ax.imshow(marginal_pp, cmap='jet', extent=(data_x.min(), data_x.max(), data_y.min(), data_y.max()),
                #           origin='lower')
                ax.imshow(ln_marginal_pp, cmap='jet',
                          extent=(left, right, bottom, top),
                          origin='lower')

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
                    ax.set_xlabel(label[col], fontsize=16)
                if col == 0:
                    ax.set_ylabel(label[row], fontsize=16)

    fig.text(0.5, 0.7, f'Truth: {truth}\n\n'
                       f'Photsys: {info[0]}\nIMF: {info[1]}\nBinary: {info[2]}\nLikelihood: {info[3]}\n'
                       f'Synthetic star number: {info[4]}',
             fontsize=16, ha='left')
    plt.savefig(savefig_path, bbox_inches='tight')
    plt.close(fig)


# %%
import joblib

logage_val = 7.
mh_val = 0.
dist_val = 780.
Av_val = 0.5
fb_val = 0.5
n_val = 1500
theta_val = logage_val, mh_val, dist_val, Av_val, fb_val

logage_step = 0.1  # 0.1
mh_step = 0.1  # 0.05
step = (logage_step, mh_step)

logage = np.arange(6.6, 8., logage_step)
mh = np.arange(-0.9, 0.7, mh_step)
dist = np.arange(750, 850, 10)  # 10
Av = np.arange(0., 1., 0.1)  # 0.1
fb = np.arange(0.2, 1., 0.1)  # 0.1
times = 2
n_stars = 5000
photsys = 'CSST'

truth = list(theta_val)
parameters = [logage, mh, dist, Av, fb]

label = ['$log_{10}{\\tau}$', '[M/H]', '$d $(kpc)', '$A_{v}$', '$f_{b}$']
info_h2h_cmd = [photsys, 'kroupa01', 'BinMRD(uniform distribution)', 'Hist2Hist4CMD(6)', n_stars]
info_h2p_cmd = [photsys, 'kroupa01', 'BinMRD(uniform distribution)', 'Hist2Point4CMD(6)', n_stars]

dir = '/home/shenyueyue/Projects/starcat/data/corner/demo_age7/'
joint_lnlike_h2h_cmd = joblib.load(dir + 'h2h_cmd(6).joblib')
joint_lnlike_h2p_cmd = joblib.load(dir + 'h2p_cmd(6).joblib')

draw_corner(truth=truth, parameters=parameters, ln_joint_distribution=joint_lnlike_h2h_cmd,
            label=label, info=info_h2h_cmd,
            savefig_path=dir + 'h2h_cmd2.png')

draw_corner(truth=truth, parameters=parameters, ln_joint_distribution=joint_lnlike_h2p_cmd,
            label=label, info=info_h2p_cmd,
            savefig_path=dir + 'h2p_cmd2.png')
