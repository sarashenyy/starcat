import matplotlib.pyplot as plt
import numpy as np

from . import config


class CMD(object):
    @staticmethod
    def extract_cmd(sample, model, photsys):
        """
        Extract color and mag from sample.

        Parameters
        ----------
        sample
        model : str
            Ioscmodel
        photsys : str
            photmetry system

        Returns
        -------

        """
        source = config.config[model][photsys]
        m = sample[source['mag']]
        c = sample[source['color'][0]] - sample[source['color'][1]]
        return c, m

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
        hist = plt.hist2d(color, mag, bins)
        h, x_edges, y_edges = hist[0], hist[1], hist[2]
        return h, x_edges, y_edges

    @staticmethod
    def hist2d_norm(color, mag, bins: int):
        # h, x_edges, y_edges = CMD.hist2d(color, mag, c_grid, m_grid)

        # h, x_edges, y_edges = CMD.hist2d(color, mag, c_bin, m_bin)
        h, x_edges, y_edges = CMD.hist2d(color, mag, bins)
        h = h / np.sum(h)
        return h, x_edges, y_edges

    @staticmethod
    def extract_hist2d(sample, model, photsys, bins):
        """
        From sample extreact hist2d.
        """
        c, m = CMD.extract_cmd(sample, model, photsys)
        # bins: int for sample_obs
        if isinstance(bins, int):
            h, x_edges, y_edges = CMD.hist2d(c, m, bins)
        # bins = (x_edges, y_edges) for cut sample_syn to sample_obs
        elif isinstance(bins, tuple):
            h, x_edges, y_edges = np.histogram2d(c, m, bins=bins)
        return h, x_edges, y_edges

    @staticmethod
    def extract_hist2d_norm(sample, model, photsys, bins):
        c, m = CMD.extract_cmd(sample, model, photsys)
        if isinstance(bins, int):
            h, x_edges, y_edges = CMD.hist2d_norm(c, m, bins)
        # bins = (x_edges, y_edges)
        elif isinstance(bins, tuple):
            h, x_edges, y_edges = np.histogram2d(c, m, bins=bins)
            h = h / np.sum(h)
        return h, x_edges, y_edges

    @staticmethod
    def plot_cmd(sample, model, photsys, bins: int):
        c, m = CMD.extract_cmd(sample, model, photsys)
        fig, ax = plt.subplots()
        ax.hist2d(c, m, bins, cmap='Blues')
        ax.invert_yaxis()
        ax.set_xlabel('Color (mag)')
        ax.set_ylabel('Mag (mag)')
        fig.show()

    @staticmethod
    def plot_resid(sample_obs, sample_syn, model, photsys, bins: int):
        h_obs, xe_obs, ye_obs = CMD.extract_hist2d(sample_obs, model, photsys, bins)
        h_syn, _, _ = CMD.extract_hist2d(sample_syn, model, photsys, bins=(xe_obs, ye_obs))
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
