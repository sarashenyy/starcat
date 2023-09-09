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
        c = sample[source['color'][0] - source['color'][1]]
        return c, m

    @staticmethod
    def hist2d(color, mag, c_grid=None, m_grid=None):
        # adaptive grid
        if m_grid is None:
            mstart = min(mag)
            mend = max(mag)
            mstep = 0.1
        elif m_grid is not None:
            mstart, mend, mstep = m_grid
        if c_grid is None:
            cstart = min(color)
            cend = max(color)
            cstep = mstep * (cend - cstart) / (mend - mstart)
        elif c_grid is not None:
            # define grid edges
            cstart, cend, cstep = c_grid
        c_bin = np.arange(cstart, cend, cstep)
        m_bin = np.arange(mstart, mend, mstep)
        h, x_edges, y_edges = np.histogram2d(color, mag, bins=(c_bin, m_bin))
        return h, x_edges, y_edges

    @staticmethod
    def hist2d_norm(color, mag, c_grid=None, m_grid=None):
        h, x_edges, y_edges = CMD.hist2d(color, mag, c_grid, m_grid)
        h = h / np.sum(h)
        return h, x_edges, y_edges

    @staticmethod
    def extract_hist2d(sample, model, photsys, c_grid=None, m_grid=None):
        c, m = CMD.extract_cmd(sample, model, photsys)
        h, x_edges, y_edges = CMD.hist2d(c, m, c_grid, m_grid)
        return h, x_edges, y_edges

    @staticmethod
    def extract_hist2d_norm(sample, model, photsys, c_grid=None, m_grid=None):
        c, m = CMD.extract_cmd(sample, model, photsys)
        h, x_edges, y_edges = CMD.hist2d_norm(c, m, c_grid, m_grid)
        return h, x_edges, y_edges
