import warnings
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from . import config
from .cmd import CMD
from .logger import log_time
from .widgets import round_to_step


class LikelihoodFunc(ABC):
    @abstractmethod
    def eval_lnlike(self, sample_obs, sample_syn):
        pass


class Hist2Hist4CMD(LikelihoodFunc):
    """
    lnlike(H_{syn},H_{obs}) = -\frac{1}{2}\sum{\frac{(H_{obs}-H_{syn})^2}{H_{obs}+H_{syn}+1}}
    """

    def __init__(self, model, photsys, bins: int, number=1):
        self.func = 'hist2hist'
        self.model = model
        self.photsys = photsys
        self.bins = bins
        self.number = number

    @log_time
    def eval_lnlike(self, sample_obs, sample_syn):
        if self.number == 1:
            h_obs, xe_obs, ye_obs = CMD.extract_hist2d(
                False, sample_obs, self.model, self.photsys, self.bins
            )
            h_syn, _, _ = CMD.extract_hist2d(
                True, sample_syn, self.model, self.photsys, bins=(xe_obs, ye_obs)
            )
            n_syn = len(sample_syn)
            n_obs = len(sample_obs)
            h_syn = h_syn / (n_syn / n_obs)
            lnlike = -0.5 * np.sum(np.square(h_obs - h_syn) / (h_obs + h_syn + 1))
            # # * NOTE correction, make max(lnlike_list)=0 !! IN corner_tests.draw_corner.py !!
            # delta = np.max(lnlike)
            # lnlike = lnlike - delta
            return lnlike

        elif self.number > 1:
            source = config.config[self.model][self.photsys]
            n_syn = len(sample_syn)
            n_obs = len(sample_obs)
            lnlikes = []
            for i in range(self.number):
                m_obs = sample_obs[source['mag'][i]]
                c_obs = sample_obs[source['color'][i][0]] - sample_obs[source['color'][i][1]]
                m_syn = sample_syn[source['mag'][i]]
                c_syn = sample_syn[source['color'][i][0]] - sample_syn[source['color'][i][1]]
                if isinstance(self.bins, int):
                    hist_obs = plt.hist2d(c_obs, m_obs, self.bins)
                    h_obs, xe_obs, ye_obs = hist_obs[0], hist_obs[1], hist_obs[2]
                    hist_syn = plt.hist2d(c_syn, m_syn, bins=(xe_obs, ye_obs))
                    h_syn, xe_syn, ye_syn = hist_syn[0], hist_syn[1], hist_syn[2]
                elif isinstance(self.bins, tuple):
                    h_obs, xe_obs, ye_obs = np.histogram2d(c_obs, m_obs, bins=self.bins)
                    h_syn, xe_syn, ye_syn = np.histogram2d(c_syn, m_syn, bins=self.bins)

                h_syn = h_syn / (n_syn / n_obs)
                aux = -0.5 * np.sum(np.square(h_obs - h_syn) / (h_obs + h_syn + 1))
                lnlikes.append(aux)
            lnlike = np.sum(lnlikes)
            # # * NOTE correction, make max(lnlike_list)=0 !! IN corner_tests.draw_corner.py !!
            # delta = np.max(lnlike)
            # lnlike = lnlike - delta
            return lnlike


# class Hist2Hist4Bands(LikelihoodFunc):
#     def __init__(self, model, photsys, **kwargs):
#         """
#
#         Parameters
#         ----------
#         model
#         photsys
#         **kwargs : dict
#             - step
#             - bins
#         """
#         self.func = 'band2band'
#         self.model = model
#         self.photsys = photsys
#         source = config.config[self.model][self.photsys]
#         self.bands = source['bands']
#         self.bands_err = source['bands_err']
#         step_input = kwargs.get('step')
#         bins_input = kwargs.get('bins')
#         self.bins_flag = False
#         self.step_flag = False
#
#         if step_input is None and bins_input is None:
#             self.bins = None
#             self.bins_flag = True
#         elif step_input is not None and bins_input is None:
#             self.step = step_input
#             self.step_flag = True
#         elif step_input is None and bins_input is not None:
#             self.bins = bins_input
#             self.bins_flag = True
#         elif step_input is not None and bins_input is not None:
#             print("WARNING! Set bins and step at the same time, change to default (Sturges Rule).")
#             self.bins_flag = True
#
#     def eval_lnlike(self, sample_obs, sample_syn):
#         """
#
#         Parameters
#         ----------
#         sample_obs
#         sample_syn
#
#         Returns
#         -------
#
#         """
#
#         # calculate for each band
#         band_lnlikes = []
#         n_syn = len(sample_syn)
#         n_obs = len(sample_obs)
#
#         for _, _err in zip(self.bands, self.bands_err):
#             band_obs = sample_obs[_]
#             # band_obs_err = sample_obs[_]
#             band_syn = sample_syn[_]
#             if self.bins_flag:
#                 if self.bins is None:
#                     self.bins = int(1 + np.log2(len(sample_obs)))
#                     band_bin = self.bins
#                 elif isinstance(self.bins, int):
#                     band_bin = self.bins
#             elif self.step_flag:
#                 band_bin = np.arange(min(band_obs), max(band_obs) + self.step, self.step)
#
#             h_obs, he_obs = np.histogram(band_obs, bins=band_bin)
#             h_syn, he_syn = np.histogram(band_syn, bins=band_bin)
#             h_syn = h_syn / (n_syn / n_obs)
#             aux = -0.5 * np.sum(np.square(h_obs - h_syn) / (h_obs + h_syn + 1))
#             band_lnlikes.append(aux)
#         lnlike = np.sum(band_lnlikes)
#         return lnlike


class Hist2Point4CMD(LikelihoodFunc):
    """
    likelihood = \prod{p_{i}^{n_i}}
    log(likelihood) =\sum{n_{i}\ln{p_{i}}}
    """

    def __init__(self, model, photsys, bins: int, number=1):
        """

        Parameters
        ----------
        model
        photsys
        bins
        number : int
            number of CMDs
        """
        self.func = 'hist2point'
        self.model = model
        self.photsys = photsys
        self.bins = bins
        self.number = number

    @log_time
    def eval_lnlike(self, sample_obs, sample_syn):
        if self.number == 1:
            h_obs, xe_obs, ye_obs = CMD.extract_hist2d(
                False, sample_obs, self.model, self.photsys, self.bins
            )
            h_syn, _, _ = CMD.extract_hist2d(
                True, sample_syn, self.model, self.photsys, bins=(xe_obs, ye_obs)
            )
            epsilon = 1e-20
            h_syn = h_syn / np.sum(h_syn)
            h_syn = h_syn + epsilon
            h_syn = h_syn / np.sum(h_syn)
            # lnlike = np.sum(h_obs * np.log10(h_syn))
            lnlike = np.sum(h_obs * np.log(h_syn))
            # # * NOTE correction, make max(lnlike_list)=0 !! IN corner_tests.draw_corner.py !!
            # delta = np.max(lnlike)
            # lnlike = lnlike - delta
            return lnlike
        elif self.number > 1:
            source = config.config[self.model][self.photsys]
            epsilon = 1e-20
            lnlikes = []
            for i in range(self.number):
                m_obs = sample_obs[source['mag'][i]]
                c_obs = sample_obs[source['color'][i][0]] - sample_obs[source['color'][i][1]]
                m_syn = sample_syn[source['mag'][i]]
                c_syn = sample_syn[source['color'][i][0]] - sample_syn[source['color'][i][1]]
                if isinstance(self.bins, int):
                    hist_obs = plt.hist2d(c_obs, m_obs, self.bins)
                    h_obs, xe_obs, ye_obs = hist_obs[0], hist_obs[1], hist_obs[2]
                    hist_syn = plt.hist2d(c_syn, m_syn, self.bins)
                    h_syn, xe_syn, ye_syn = hist_syn[0], hist_syn[1], hist_syn[2]
                elif isinstance(self.bins, tuple):
                    h_obs, xe_obs, ye_obs = np.histogram2d(c_obs, m_obs, bins=self.bins)
                    h_syn, xe_syn, ye_syn = np.histogram2d(c_syn, m_syn, bins=self.bins)

                h_syn = h_syn / np.sum(h_syn)
                h_syn = h_syn + epsilon
                h_syn = h_syn / np.sum(h_syn)
                aux = np.sum(h_obs * np.log(h_syn))
                lnlikes.append(aux)
            lnlike = np.sum(lnlikes)
            # # * NOTE correction, make max(lnlike_list)=0 !! IN corner_tests.draw_corner.py !!
            # delta = np.max(lnlike)
            # lnlike = lnlike - delta
            return lnlike


def lnlike_2p(theta_age_mh, fb, dm, step, isoc, likelihoodfunc, synstars, n_stars, sample_obs):
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
    lnlike = lnlike_5p(theta, step, isoc, likelihoodfunc, synstars, n_stars, sample_obs)
    return lnlike


@log_time
def lnlike_5p(theta, step, isoc, likelihoodfunc, synstars, n_stars, sample_obs, times=1):
    # try:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    logage, mh, dm, Av, fb = theta
    logage_step, mh_step = step
    logage = round_to_step(logage, logage_step)
    mh = round_to_step(mh, mh_step)
    theta = logage, mh, dm, Av, fb
    # !NOTE: theta range, dm(LMC,SMC~18.5, M31~24.5)
    # !      Av(for M31) range [0, 3] Fouesneau2014(https://iopscience.iop.org/article/10.1088/0004-637X/786/2/117)
    # !                               Li Lu MIMO & PhD thesis
    # !      dm(for Gaia) range [3, 15] Li Lu MIMO & PhD thesis
    # * Note [M/H] range in [-2, 0.7]? Dias2021 from [-0.9, 0.7]
    if ((logage > 10.0) or (logage < 6.7) or (mh < -2.) or (mh > 0.4) or
            (dm < 3.) or (dm > 15.) or (Av < 0.) or (Av > 3.) or (fb < 0.2) or (fb > 1.)):
        return -np.inf
    else:
        if times == 1:
            sample_syn = synstars(theta, n_stars, isoc, logage_step=logage_step, mh_step=mh_step)
            if sample_syn is False:
                # return 1e10
                return -np.inf
            else:
                lnlike = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
                return lnlike
            # lnlike = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
            # return lnlike

        elif times > 1:
            # * without acceleration
            lnlike_list = []
            for i in range(times):
                sample_syn = synstars(theta, n_stars, isoc, logage_step=logage_step, mh_step=mh_step)
                # if sample_syn is False:
                #     lnlike_one = 1e10
                # else:
                #     lnlike_one = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
                lnlike_one = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
                lnlike_list.append(lnlike_one)
            lnlike = np.sum(lnlike_list) / times

            # * acceleration with parallelization
            # /home/shenyueyue/Packages/miniconda3/envs/mcmc/lib/python3.10/site-packages/joblib/externals/loky/backend
            # /resource_tracker.py:318: UserWarning: resource_tracker:
            # There appear to be 278 leaked semlock objects to clean up at shutdown
            #   warnings.warn('resource_tracker: There appear to be %d '

            # def compute_lnlike_one_iteration(i):
            #     sample_syn = synstars(theta, isoc, logage_step=logage_step, mh_step=mh_step)
            #     lnlike_one = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
            #     return lnlike_one
            #
            # lnlike_list = Parallel(n_jobs=-1)(delayed(compute_lnlike_one_iteration)(i) for i in range(times))
            # # lnlike_list = Parallel(n_jobs=-1, temp_folder='/home/shenyueyue/Projects/starcat/temp_folder')(
            # #     delayed(compute_lnlike_one_iteration)(i) for i in range(times))
            # lnlike = np.sum(lnlike_list) / times

            return lnlike
    # except ZeroDivisionError as e:
    #     # Handle the division by zero error here
    #     print(f"!!!!ZeroDivisionError occurred: {e}")
    #     print(theta)
    #     # You can add custom handling or return a specific value as needed
    #     return None  # Or any value that makes sense in your context
    # import warnings
    # warnings.filterwarnings("ignore", category=RuntimeWarning)
    # try:
    #     sample_syn = synstars(theta, step, isoc)
    #     lnlike = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
    #     return lnlike
    # except RuntimeWarning:
    #     print(theta)
    #     return -np.inf
