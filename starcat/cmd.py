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
    def hist2d(color, mag, c_bin: int, m_bin: int):
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
        cstart, cend = min(color), max(color)
        mstart, mend = min(mag), max(mag)
        c_bins = np.linspace(cstart, cend, c_bin + 1)
        m_bins = np.linspace(mstart, mend, m_bin + 1)
        h, x_edges, y_edges = np.histogram2d(color, mag, bins=(c_bins, m_bins))
        return h, x_edges, y_edges

    @staticmethod
    def hist2d_norm(color, mag, c_bin: int, m_bin: int):
        # h, x_edges, y_edges = CMD.hist2d(color, mag, c_grid, m_grid)
        h, x_edges, y_edges = CMD.hist2d(color, mag, c_bin, m_bin)
        h = h / np.sum(h)
        return h, x_edges, y_edges

    @staticmethod
    def extract_hist2d(sample, model, photsys, c_bin: int, m_bin: int):
        """
        From sample extreact hist2d.
        """
        c, m = CMD.extract_cmd(sample, model, photsys)
        h, x_edges, y_edges = CMD.hist2d(c, m, c_bin, m_bin)
        return h, x_edges, y_edges

    @staticmethod
    def extract_hist2d_norm(sample, model, photsys, c_bin: int, m_bin: int):
        c, m = CMD.extract_cmd(sample, model, photsys)
        h, x_edges, y_edges = CMD.hist2d_norm(c, m, c_bin, m_bin)
        return h, x_edges, y_edges
