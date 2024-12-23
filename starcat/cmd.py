import time

import matplotlib.pyplot as plt
import numpy as np
from robustgp import ITGP

from . import config


class CMD(object):
    @staticmethod
    def extract_cmd(sample, model, photsys, synthetic):
        """
        Extract color and mag from sample.

        Parameters
        ----------
        synthetic
        sample
        model : str
            Ioscmodel
        photsys : str
            photmetry system

        Returns
        -------

        """
        if synthetic is True:
            source = config.config[model][photsys]
        elif synthetic is False:
            source = config.config['observation'][photsys]

        m = sample[source['mag'][0]].to_numpy()
        c = (sample[source['color'][0][0]] - sample[source['color'][0][1]]).to_numpy()
        return c, m

    @staticmethod
    def get_membership_prob(sample, model, photsys):
        source = config.config['observation'][photsys]
        prob = sample[source['prob']].to_numpy()
        return prob

    @staticmethod
    def find_rigdeline(color, mag):
        print('start robustgp to find rigde line...')
        st = time.time()
        res = ITGP(mag, color,
                   alpha1=0.5, alpha2=0.975, nsh=2, ncc=2, nrw=1,
                   optimize_kwargs=dict(optimizer='lbfgsb')
                   )
        gp, consistency = res.gp, res.consistency

        RL_m = np.linspace(np.min(mag), np.max(mag), num=200)
        RL_c, _ = gp.predict(RL_m.reshape(-1, 1))
        RL_c = RL_c.ravel()
        ed = time.time()
        print(f'finish robustgp in {ed - st:.2f}s')
        return RL_c, RL_m

    @staticmethod
    def extract_error(sample, model, photsys, synthetic):
        if synthetic is True:
            source = config.config[model][photsys]
        elif synthetic is False:
            source = config.config['observation'][photsys]

        m_err = sample[source['mag_err'][0]].to_numpy()
        c1_err = sample[source['color_err'][0][0]].to_numpy()
        c2_err = sample[source['color_err'][0][1]].to_numpy()
        c_err = np.sqrt(c1_err ** 2 + c2_err ** 2)
        return c_err, m_err

    @staticmethod
    def hist2d(color, mag, bins: int):
        # adaptive grid
        # if c_grid is None:
        #     cstart = min(color)
        #     cend = max(color)
        #     cstep = 0.1
        # elif c_grid is not None:
        #     # define grid edges
        #     cstart, cend, cstep = c_grid
        # if m_grid is None:
        #     mstart = min(mag)
        #     mend = max(mag)
        #     mstep = cstep * (mend - mstart) / (cend - cstart)
        # elif m_grid is not None:
        #     mstart, mend, mstep = m_grid
        # c_bin = np.arange(cstart, cend, cstep)
        # m_bin = np.arange(mstart, mend, mstep)
        # h, x_edges, y_edges = np.histogram2d(color, mag, bins=(c_bin, m_bin))

        # cstart, cend = min(color), max(color)
        # mstart, mend = min(mag), max(mag)
        # c_bins = np.linspace(cstart, cend, c_bin + 1)
        # m_bins = np.linspace(mstart, mend, m_bin + 1)
        # h, x_edges, y_edges = np.histogram2d(color, mag, bins=(c_bins, m_bins))
        h, x_edges, y_edges = np.histogram2d(color, mag, bins=bins)
        return h, x_edges, y_edges

    @staticmethod
    def hist2d_norm(color, mag, bins: int):
        # h, x_edges, y_edges = CMD.hist2d(color, mag, c_grid, m_grid)

        # h, x_edges, y_edges = CMD.hist2d(color, mag, c_bin, m_bin)
        h, x_edges, y_edges = CMD.hist2d(color, mag, bins)
        h = h / np.sum(h)
        return h, x_edges, y_edges

    @staticmethod
    def extract_hist2d(synthetic, sample, model, photsys, bins):
        """
        From sample extreact hist2d.
        """
        c, m = CMD.extract_cmd(sample, model, photsys, synthetic)
        # bins: int for sample_obs
        if isinstance(bins, int):
            h, x_edges, y_edges = CMD.hist2d(c, m, bins)
        # bins = (x_edges, y_edges) for cut sample_syn to sample_obs
        elif isinstance(bins, tuple):
            h, x_edges, y_edges = np.histogram2d(c, m, bins=bins)
        return h, x_edges, y_edges

    @staticmethod
    def extract_hist2d_norm(synthetic, sample, model, photsys, bins):
        c, m = CMD.extract_cmd(sample, model, photsys, synthetic)
        if isinstance(bins, int):
            h, x_edges, y_edges = CMD.hist2d_norm(c, m, bins)
        # bins = (x_edges, y_edges)
        elif isinstance(bins, tuple):
            h, x_edges, y_edges = np.histogram2d(c, m, bins=bins)
            h = h / np.sum(h)
        return h, x_edges, y_edges

    @staticmethod
    def plot_cmd(synthetic, sample, model, photsys, bins: int):
        c, m = CMD.extract_cmd(sample, model, photsys, synthetic)
        fig, ax = plt.subplots()
        ax.hist2d(c, m, bins, cmap='Blues')
        ax.invert_yaxis()
        ax.set_xlabel('Color (mag)')
        ax.set_ylabel('Mag (mag)')
        fig.show()

    @staticmethod
    def plot_resid(synthetic, sample_obs, sample_syn, model, photsys, bins: int):
        h_obs, xe_obs, ye_obs = CMD.extract_hist2d(sample_obs, model, photsys, synthetic, bins)
        h_syn, _, _ = CMD.extract_hist2d(sample_syn, model, photsys, synthetic, bins=(xe_obs, ye_obs))
        residuals = h_obs / np.sum(h_obs) - h_syn / np.sum(h_syn)

        # 绘制残差图
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            residuals, cmap='bwr', origin='lower',
            # extent=[xe_obs[0], xe_obs[-1], ye_obs[0], ye_obs[-1]],
            aspect='equal'
        )
        cbar = fig.colorbar(im, ax=ax, label='Residuals')
        ax.set_xlabel('Color')
        ax.set_ylabel('Magnitude')
        ax.set_title('Histogram2D Residuals')
        ax.invert_yaxis()
        fig.show()

        return residuals, xe_obs, ye_obs
