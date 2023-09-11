import warnings
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

    def __init__(self, model, photsys, bins: int):
        self.func = 'hist2hist'
        self.model = model
        self.photsys = photsys
        self.bins = bins

    def eval_lnlike(self, sample_obs, sample_syn):
        h_obs, _, _ = CMD.extract_hist2d(sample_obs, self.model, self.photsys, self.bins)
        h_syn, _, _ = CMD.extract_hist2d(sample_syn, self.model, self.photsys, self.bins)
        n_syn = len(sample_syn)
        n_obs = len(sample_obs)
        h_syn = h_syn / (n_syn / n_obs)
        lnlike = -0.5 * np.sum(np.square(h_obs - h_syn) / (h_obs + h_syn + 1))
        # !NOTE correction is max(lnlike) in param space
        correction = -230
        lnlike = lnlike - correction
        return lnlike


class Hist2Point(LikelihoodFunc):
    """
    likelihood = \prod{p_{i}^{n_i}}
    log(likelihood) =\sum{n_{i}\ln{p_{i}}}
    """

    def __init__(self, model, photsys, bins: int):
        self.func = 'hist2point'
        self.model = model
        self.photsys = photsys
        self.bins = bins

    def eval_lnlike(self, sample_obs, sample_syn):
        h_obs, _, _ = CMD.extract_hist2d(sample_obs, self.model, self.photsys, self.bins)
        h_syn, _, _ = CMD.extract_hist2d(sample_syn, self.model, self.photsys, self.bins)
        epsilon = 1e-20
        h_syn = h_syn / np.sum(h_syn)
        h_syn = h_syn + epsilon
        h_syn = h_syn / np.sum(h_syn)
        lnlike = np.sum(h_obs * np.log10(h_syn))
        # !NOTE correction is max(lnlike) in param space
        correction = -4100
        lnlike = lnlike - correction
        return lnlike


def lnlike_2p(theta_age_mh, fb, dm, step, isoc, likelihoodfunc, synstars, sample_obs):
    """

    Parameters
    ----------
    synstars : starcat.SynStars
        the instantiated SynStars()
    likelihoodfunc : subclass
        subclass of LikelihoodFunc : Hist2Hist(), Hist2Point()
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
    lnlike = lnlike_4p(theta, step, isoc, likelihoodfunc, synstars, sample_obs)
    return lnlike


def lnlike_4p(theta, step, isoc, likelihoodfunc, synstars, sample_obs):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    logage, mh, fb, dm = theta
    if (logage > 10.0) or (logage < 6.6) or (mh < -0.9) or (mh > 0.7) or (fb < 0.05) or (fb > 1) or (dm < 2) or (
            dm > 20):
        return -np.inf
    sample_syn = synstars(theta, step, isoc)
    lnlike = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
    return lnlike
    # import warnings
    # warnings.filterwarnings("ignore", category=RuntimeWarning)
    # try:
    #     sample_syn = synstars(theta, step, isoc)
    #     lnlike = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
    #     return lnlike
    # except RuntimeWarning:
    #     print(theta)
    #     return -np.inf
