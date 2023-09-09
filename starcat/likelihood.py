from abc import ABC, abstractmethod

import numpy as np

from .cmd import CMD


class LikelihoodFunc(ABC):
    @abstractmethod
    def eval_lnlike(self, sample_obs, sample_syn):
        pass


class Hist2Hist(LikelihoodFunc):
    """
    lnlike(H_{syn},H_{obs}) = -\frac{1}{2}\sum{\frac{(H_{obs}-H_{syn})^2}{H_{obs}+H_{syn}+1}}
    """

    def __init__(self, model, photsys, c_grid=None, m_grid=None):
        self.func = 'hist2hist'
        self.model = model
        self.photsys = photsys
        self.c_grid = c_grid
        self.m_grid = m_grid

    def eval_lnlike(self, sample_obs, sample_syn):
        h_obs, _, _ = CMD.extract_hist2d(sample_obs, self.model, self.photsys, self.c_grid, self.m_grid)
        h_syn, _, _ = CMD.extract_hist2d(sample_syn, self.model, self.photsys, self.c_grid, self.m_grid)
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

    def __init__(self, model, photsys, c_grid=None, m_grid=None):
        self.func = 'hist2point'
        self.model = model
        self.photsys = photsys
        self.c_grid = c_grid
        self.m_grid = m_grid

    def eval_lnlike(self, sample_obs, sample_syn):
        h_obs, _, _ = CMD.extract_hist2d(sample_obs, self.model, self.photsys, self.c_grid, self.m_grid)
        h_syn, _, _ = CMD.extract_hist2d(sample_syn, self.model, self.photsys, self.c_grid, self.m_grid)
        epsilon = 1e-20
        h_syn = h_syn / np.sum(h_syn)
        h_syn = h_syn + epsilon
        h_syn = h_syn / np.sum(h_syn)
        lnlike = np.sum(h_obs * np.log10(h_syn))
        return lnlike


class Likelihood(object):
    def __init__(self, likelihoodfunc, synstars):
        """

        Parameters
        ----------
        synstars : starcat.SynStars
            the instantiated SynStars()
        likelihoodfunc : subclass
            subclass of LikelihoodFunc : Hist2Hist(), Hist2Point()

        """
        self.likelihoodfunc = likelihoodfunc
        self.synstars = synstars

    def lnlike_2p(self, theta_age_mh, fb, dm, step, isoc, sample_obs):
        """

        Parameters
        ----------
        sample_obs : pd.DataFrame
        isoc : starcat.Isoc()
            the instantiated Isoc().
            ```python
            p = Parsec()
            i = Isoc(p)
            ```
        theta_age_mh : tuple
            logage, mh
        fb : float
        dm : float
        step : tuples
            logage_step, mh_step

        Returns
        -------

        """
        logage, mh = theta_age_mh
        theta = logage, mh, fb, dm
        if (logage > 10.0) or (logage < 6.6) or (mh < -0.9) or (mh > 0.7) or (fb < 0.05) or (fb > 1) or (dm < 2) or (
                dm > 20):
            return -np.inf
        sample_syn = self.synstars(theta, step, isoc)
        lnlike = self.likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
        return lnlike

    def lnlike_4p(self, theta, step, isoc, sample_obs):
        logage, mh, fb, dm = theta
        if (logage > 10.0) or (logage < 6.6) or (mh < -0.9) or (mh > 0.7) or (fb < 0.05) or (fb > 1) or (dm < 2) or (
                dm > 20):
            return -np.inf
        sample_syn = self.synstars(theta, step, isoc)
        lnlike = self.likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
        return lnlike
