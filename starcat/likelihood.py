import warnings
from abc import ABC, abstractmethod

import numpy as np
from joblib import Parallel, delayed

from .cmd import CMD
from .logger import log_time


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

    @log_time
    def eval_lnlike(self, sample_obs, sample_syn):
        h_obs, xe_obs, ye_obs = CMD.extract_hist2d(sample_obs, self.model, self.photsys, self.bins)
        h_syn, _, _ = CMD.extract_hist2d(sample_syn, self.model, self.photsys, bins=(xe_obs, ye_obs))
        n_syn = len(sample_syn)
        n_obs = len(sample_obs)
        h_syn = h_syn / (n_syn / n_obs)
        lnlike = -0.5 * np.sum(np.square(h_obs - h_syn) / (h_obs + h_syn + 1))
        # !NOTE correction is max(lnlike) in param space
        # correction = -230
        # lnlike = lnlike - correction - 180
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

    @log_time
    def eval_lnlike(self, sample_obs, sample_syn):
        h_obs, xe_obs, ye_obs = CMD.extract_hist2d(sample_obs, self.model, self.photsys, self.bins)
        h_syn, _, _ = CMD.extract_hist2d(sample_syn, self.model, self.photsys, bins=(xe_obs, ye_obs))
        epsilon = 1e-20
        h_syn = h_syn / np.sum(h_syn)
        h_syn = h_syn + epsilon
        h_syn = h_syn / np.sum(h_syn)
        # lnlike = np.sum(h_obs * np.log10(h_syn))
        lnlike = np.sum(h_obs * np.log(h_syn))
        # !NOTE correction is max(lnlike) in param space
        # correction = -4100
        # lnlike = lnlike - correction - 1960
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
    lnlike = lnlike_5p(theta, step, isoc, likelihoodfunc, synstars, sample_obs)
    return lnlike


@log_time
def lnlike_5p(theta, step, isoc, likelihoodfunc, synstars, sample_obs, times):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    logage, mh, dist, Av, fb = theta
    logage_step, mh_step = step
    # !NOTE: theta range, dist(for M31) range [700,800] kpc
    # !      Av(for M31) range [0, 3] Fouesneau2014(https://iopscience.iop.org/article/10.1088/0004-637X/786/2/117)
    if ((logage > 10.0) or (logage < 6.6) or (mh < -0.9) or (mh > 0.7) or
            (dist < 700) or (dist > 850) or (Av < 0.) or (Av > 3.) or (fb < 0.2) or (fb > 1)):
        return -np.inf

    # sample_syn = synstars(theta, isoc, logage_step=logage_step, mh_step=mh_step)
    # lnlike = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)

    # * without acceleration
    # lnlike_list = []
    # for i in range(times):
    #     sample_syn = synstars(theta, isoc, logage_step=logage_step, mh_step=mh_step)
    #     lnlike_one = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
    #     lnlike_list.append(lnlike_one)
    # lnlike = np.sum(lnlike_list) / times

    # * acceleration with parallelization
    def compute_lnlike_one_iteration(i):
        sample_syn = synstars(theta, isoc, logage_step=logage_step, mh_step=mh_step)
        lnlike_one = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
        return lnlike_one

    lnlike_list = Parallel(n_jobs=-1)(delayed(compute_lnlike_one_iteration)(i) for i in range(times))
    # lnlike_list = Parallel(n_jobs=-1, temp_folder='/home/shenyueyue/Projects/starcat/temp_folder')(
    #     delayed(compute_lnlike_one_iteration)(i) for i in range(times))
    lnlike = np.sum(lnlike_list) / times

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
