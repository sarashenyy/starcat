from abc import ABC, abstractmethod

import numpy as np

from .cmd import CMD


class LikelihoodFunc(ABC):
    @abstractmethod
    def eval_lnlike(self, sample_obs, sample_syn, model, photsys, c_grid=None, m_grid=None):
        pass


class Hist2Hist(LikelihoodFunc):
    """
    lnlike(H_{syn},H_{obs}) = -\frac{1}{2}\sum{\frac{(H_{obs}-H_{syn})^2}{H_{obs}+H_{syn}+1}}
    """

    def __init__(self):
        self.func = 'hist2hist'

    def eval_lnlike(self, sample_obs, sample_syn, model, photsys, c_grid=None, m_grid=None):
        h_obs, _, _ = CMD.extract_hist2d(sample_obs, model, photsys, c_grid, m_grid)
        h_syn, _, _ = CMD.extract_hist2d(sample_syn, model, photsys, c_grid, m_grid)
        n_syn = len(sample_syn)
        n_obs = len(sample_obs)
        h_syn = h_syn / (n_syn / n_obs)
        lnlike = -0.5 * np.sum(np.square(h_obs - h_syn) / (h_obs + h_syn + 1))
        return lnlike


class Hist2Point(LikelihoodFunc):
    """
    likelihood = \prod{p_{i}^{n_i}}
    log(likelihood) =\sum{n_{i}\ln{p_{i}}}
    """

    def __init__(self):
        self.func = 'hist2point'

    def eval_lnlike(self, sample_obs, sample_syn, model, photsys, c_grid=None, m_grid=None):
        h_obs, _, _ = CMD.extract_hist2d(sample_obs, model, photsys, c_grid, m_grid)
        h_syn, _, _ = CMD.extract_hist2d(sample_syn, model, photsys, c_grid, m_grid)
        epsilon = 1e-20
        h_syn = h_syn / np.sum(h_syn)
        h_syn = h_syn + epsilon
        h_syn = h_syn / np.sum(h_syn)
        lnlike = np.sum(h_obs * np.log10(h_syn))
        return lnlike
