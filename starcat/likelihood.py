import warnings
from abc import ABC, abstractmethod

import math
import numpy as np
from astropy.stats import knuth_bin_width, bayesian_blocks
from math import comb
from matplotlib import pyplot as plt
from scipy.stats import energy_distance, gaussian_kde

from . import config
from .cmd import CMD
from .widgets import round_to_step


class LikelihoodFunc(ABC):
    @abstractmethod
    def eval_lnlike(self, sample_obs, sample_syn):
        pass

    def get_funcname(self):
        pass


class Hist2Hist4CMD(LikelihoodFunc):
    """
    lnlike(H_{syn},H_{obs}) = -\frac{1}{2}\sum{\frac{(H_{obs}-H_{syn})^2}{H_{obs}+H_{syn}+1}}
    """

    def __init__(self,
                 model,
                 photsys,
                 bin_method,
                 number=1,
                 **kwargs):
        self.func = 'hist2hist'
        self.model = model
        self.photsys = photsys
        self.number = number
        self.bin_method = bin_method
        if self.bin_method == 'fixed':
            self.bins = kwargs.get('bins')

    # @log_time
    def eval_lnlike(self, sample_obs, sample_syn):
        if self.number == 1:
            h_syn = None
            h_obs = None
            if self.bin_method == 'fixed':
                h_obs, h_syn = bin_fixed(sample_obs, sample_syn, self.model, self.photsys, self.bins)

            elif self.bin_method == 'knuth':
                h_obs, h_syn = bin_knuth(sample_obs, sample_syn, self.model, self.photsys)

            elif self.bin_method == 'blocks':
                h_obs, h_syn = bin_blocks(sample_obs, sample_syn, self.model, self.photsys)
            # TODO: 修改n_syn，此时的长度应该是在sample_obs数据范围内的长度
            # TODO：同时增加对CSST的判断逻辑，对CSST，sample_syn中包含极限星等以下的数据
            # # 画图，显示h_obs与h_syn
            # from matplotlib.colors import LinearSegmentedColormap
            # colors_blue = [(1, 1, 1), (0, 0, 1)]
            # cmap_blue = LinearSegmentedColormap.from_list('custom_cmap_blue', colors_blue)
            #
            # # 定义白色到红色渐变colormap
            # colors_red = [(1, 1, 1), (1, 0, 0)]
            # cmap_red = LinearSegmentedColormap.from_list('custom_cmap_red', colors_red)
            # mask_syn = np.ma.masked_where(h_syn == 0, h_syn)
            # fig, ax = plt.subplots(figsize=(6.5, 4))  # 调整画布大小
            # mask_syn = np.ma.masked_where(h_syn == 0, h_syn)
            # mask_obs = np.ma.masked_where(h_obs == 0, h_obs)
            # # 绘制 h_obs
            # im_obs = ax.imshow(h_obs.T, interpolation='nearest', aspect='auto',
            #                    extent=[xe_obs[0], xe_obs[-1], ye_obs[-1], ye_obs[0]], cmap=cmap_blue)
            # cbar_obs = fig.colorbar(im_obs, ax=ax, label='Count (Observed)')
            #
            # # 绘制 h_syn
            # im_syn = ax.imshow(mask_syn.T, interpolation='nearest', aspect='auto',
            #                    extent=[xe_obs[0], xe_obs[-1], ye_obs[-1], ye_obs[0]], cmap=cmap_red)
            # cbar_syn = fig.colorbar(im_syn, ax=ax, label='Count (Synthetic)')
            #
            # plt.show()
            if h_syn is None or np.sum(h_syn) == 0:
                return -np.inf
            else:
                n_syn = len(sample_syn)  # 我发现了盲点！是盲点吗？？
                n_obs = len(sample_obs)
                h_syn = h_syn / (n_syn / n_obs)

                # # 画残差
                # resid = h_obs - h_syn
                # import matplotlib.colors as mcolors
                # norm = mcolors.TwoSlopeNorm(vmin=resid.min(), vmax=resid.max(), vcenter=0)
                # fig, ax = plt.subplots(figsize=(5, 4))  # 调整画布大小
                # # 绘制 h_obs
                # c_obs = sample_obs['BPmag'] - sample_obs['RPmag']
                # m_obs = sample_obs['Gmag']
                # top_idx = np.argmin(m_obs)
                # bottom_idx = np.argmax(m_obs)
                # left_idx = np.argmin(c_obs)
                # right_idx = np.argmax(c_obs)
                #
                # ax.scatter(c_obs, m_obs, s=1, color='grey', alpha=0.5)
                # ax.scatter(c_obs[top_idx], m_obs[top_idx], marker='*', color='red', s=50)
                # ax.scatter(c_obs[bottom_idx], m_obs[bottom_idx], marker='*', color='red', s=50)
                # ax.scatter(c_obs[left_idx], m_obs[left_idx], marker='*', color='red', s=50)
                # ax.scatter(c_obs[right_idx], m_obs[right_idx], marker='*', color='red', s=50)
                #
                # delta_x = (xe_obs[-1] - xe_obs[0]) * 0.05  # 5%空白
                # delta_y = (ye_obs[-1] - ye_obs[0]) * 0.05  # 5%空白
                # ax.set_xlim(xe_obs[0] - delta_x, xe_obs[-1] + delta_x)
                # ax.set_ylim(ye_obs[-1] + delta_y, ye_obs[0] - delta_y)
                #
                # im_obs = ax.imshow(resid.T, interpolation='nearest', aspect='auto',
                #                    extent=[xe_obs[0], xe_obs[-1], ye_obs[-1], ye_obs[0]],
                #                    cmap=plt.cm.RdBu, norm=norm)  # extent=[xe_obs[0], xe_obs[-1], ye_obs[-1], ye_obs[0]]
                # cbar_obs = fig.colorbar(im_obs, ax=ax, label='Count (resid)')
                # plt.show()

                lnlike = -0.5 * np.sum(np.square(h_obs - h_syn) / (h_obs + h_syn + 1))
                # lnlike = -(0.5 / n_obs) * np.sum(np.square(h_obs - h_syn) / (h_obs + h_syn + 1))
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

    def get_funcname(self):
        funcname = self.func
        return funcname


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
class SahaW(LikelihoodFunc):
    def __init__(self,
                 model,
                 photsys,
                 bin_method,
                 **kwargs):
        self.func = 'SahaWstatistic'
        self.model = model
        self.photsys = photsys
        self.bin_method = bin_method
        if self.bin_method == 'fixed':
            self.bins = kwargs.get('bins')

    def eval_lnlike(self, sample_obs, sample_syn):
        h_syn = None
        h_obs = None
        if self.bin_method == 'fixed':
            h_obs, h_syn = bin_fixed(sample_obs, sample_syn, self.model, self.photsys, self.bins)

        elif self.bin_method == 'knuth':
            c_obs, m_obs, c_syn, m_syn = get_CMD(sample_obs, sample_syn, self.model, self.photsys)
            h_obs, h_syn = bin_knuth(c_obs, m_obs, c_syn, m_syn)

        elif self.bin_method == 'blocks':
            c_obs, m_obs, c_syn, m_syn = get_CMD(sample_obs, sample_syn, self.model, self.photsys)
            h_obs, h_syn = bin_blocks(c_obs, m_obs, c_syn, m_syn)

        if h_syn is None or np.sum(h_syn) == 0:
            return -np.inf
        else:
            arr_obs = h_obs.ravel()
            arr_syn = h_syn.ravel()
            arr_obs = np.round(arr_obs).astype(int)
            arr_syn = np.round(arr_syn).astype(int)

            logw_list = []
            for i in range(len(arr_obs)):
                temp_c = comb(arr_obs[i] + arr_syn[i], arr_obs[i])
                temp_logc = math.log10(temp_c)
                logw_list.append(temp_logc)

        # loglike = -sum(logw_list)
        w = sum(logw_list)
        loglike = -0.01 * w
        return loglike

    def get_funcname(self):
        funcname = self.func
        return funcname


class EnergyDistance(LikelihoodFunc):
    def __init__(self,
                 model,
                 photsys,
                 bin_method,
                 **kwargs):
        self.func = 'EnergyDistance'
        self.model = model
        self.photsys = photsys
        self.bin_method = bin_method
        if self.bin_method == 'fixed':
            self.bins = kwargs.get('bins')

    def eval_lnlike(self, sample_obs, sample_syn):
        h_syn = None
        h_obs = None
        if self.bin_method == 'fixed':
            h_obs, h_syn = bin_fixed(sample_obs, sample_syn, self.model, self.photsys, self.bins)

        elif self.bin_method == 'knuth':
            c_obs, m_obs, c_syn, m_syn = get_CMD(sample_obs, sample_syn, self.model, self.photsys)
            h_obs, h_syn = bin_knuth(c_obs, m_obs, c_syn, m_syn)

        elif self.bin_method == 'blocks':
            c_obs, m_obs, c_syn, m_syn = get_CMD(sample_obs, sample_syn, self.model, self.photsys)
            h_obs, h_syn = bin_blocks(c_obs, m_obs, c_syn, m_syn)

        if h_syn is None or np.sum(h_syn) == 0:
            return -np.inf
        else:
            h_obs = h_obs / np.sum(h_obs)
            h_syn = h_syn / np.sum(h_syn)
            arr_obs = h_obs.ravel()
            arr_syn = h_syn.ravel()
            dis = energy_distance(arr_obs, arr_syn)
            lnlike = -1. * dis
            return lnlike

    def get_funcname(self):
        funcname = self.func
        return funcname


class DeltaCMD(LikelihoodFunc):

    def __init__(self,
                 model,
                 photsys,
                 bin_method,
                 **kwargs):
        self.func = 'DeltaCMD'
        self.model = model
        self.photsys = photsys
        self.bin_method = bin_method
        if self.bin_method == 'fixed':
            self.bins = kwargs.get('bins')

    def eval_lnlike(self, sample_obs, sample_syn):
        source = config.config[self.model][self.photsys]
        m_obs = sample_obs[source['mag']].values.ravel()
        dc_obs = sample_obs['delta_c'].values.ravel()
        m_syn = sample_syn[source['mag']].values.ravel()
        dc_syn = sample_syn['delta_c'].values.ravel()

        h_syn = None
        h_obs = None
        if self.bin_method == 'knuth':
            h_obs, h_syn = bin_knuth(dc_obs, m_obs, dc_syn, m_syn)

        elif self.bin_method == 'blocks':
            h_obs, h_syn = bin_blocks(dc_obs, m_obs, dc_syn, m_syn)

        if h_syn is None or np.sum(h_syn) == 0:
            return -np.inf
        else:
            # h_syn = h_syn / np.sum(h_syn)
            epsilon = 1e-20
            h_syn = h_syn + epsilon
            h_syn = h_syn / np.sum(h_syn)
            # lnlike = np.sum(h_obs * np.log10(h_syn))
            # lnlike = np.sum(h_obs * np.log(h_syn))
            # ! 增加 H_obs 归一化，是否正确？这样的话，似然函数的大小不会受到 Nstar 的影响；也能减小似然的整体数值
            # ! 如果不把 H_obs 也归一化，lnlike负太大，导致 exp(-lnlike)=0, 是均为0！
            h_obs = h_obs / np.sum(h_obs)

            #  Hsyn 归一化到 Hobs, 返回 nan
            # n_syn = len(sample_syn)  # 我发现了盲点！
            # n_obs = len(sample_obs)
            # h_syn = h_syn / (n_syn / n_obs)

            lnlike = np.sum(h_obs * np.log(h_syn))
            # # * NOTE correction, make max(lnlike_list)=0 !! IN corner_tests.draw_corner.py !!
            # delta = np.max(lnlike)
            # lnlike = lnlike - delta
            return lnlike

    def get_funcname(self):
        funcname = self.func
        return funcname


class GaussianKDE(LikelihoodFunc):
    def __init__(self,
                 model,
                 photsys,
                 # bin_method,
                 **kwargs):
        self.func = 'GaussianKDE'
        self.model = model
        self.photsys = photsys
        # self.bin_method = bin_method
        # if self.bin_method == 'fixed':
        #     self.bins = kwargs.get('bins')

    def get_funcname(self):
        funcname = self.func
        return funcname

    def eval_lnlike(self, sample_obs, sample_syn):
        c_obs, m_obs, c_syn, m_syn = get_CMD(sample_obs, sample_syn, self.model, self.photsys)
        syn_data = np.vstack((c_syn, m_syn))
        kde = gaussian_kde(syn_data)

        obs_data = np.vstack((c_obs, m_obs))
        pdf_obs = kde(obs_data)
        lnlike = np.sum(np.log(pdf_obs)) / 1150
        # n=1500 : 1150
        lnlike = lnlike + 1150
        return lnlike


class Hist2Point4CMD(LikelihoodFunc):
    """
    likelihood = \prod{p_{i}^{n_i}}
    log(likelihood) =\sum{n_{i}\ln{p_{i}}}
    """

    def __init__(self,
                 model,
                 photsys,
                 # bins: int,
                 bin_method,
                 number=1,
                 **kwargs):
        """

        Parameters
        ----------
        model
        photsys
        bins
        number : int
            number of CMDs
        kwargs :
            'bins'
        """
        self.func = 'Hist2Point'
        self.model = model
        self.photsys = photsys
        # self.bins = bins
        self.bin_method = bin_method
        self.number = number
        if self.bin_method == 'fixed':
            self.bins = kwargs.get('bins')

    # @log_time
    def eval_lnlike(self, sample_obs, sample_syn):
        if self.number == 1:
            h_syn = None
            h_obs = None
            if self.bin_method == 'fixed':
                h_obs, h_syn = bin_fixed(sample_obs, sample_syn, self.model, self.photsys, self.bins)

            elif self.bin_method == 'knuth':
                c_obs, m_obs, c_syn, m_syn = get_CMD(sample_obs, sample_syn, self.model, self.photsys)
                h_obs, h_syn = bin_knuth(c_obs, m_obs, c_syn, m_syn)

            elif self.bin_method == 'blocks':
                c_obs, m_obs, c_syn, m_syn = get_CMD(sample_obs, sample_syn, self.model, self.photsys)
                h_obs, h_syn = bin_blocks(c_obs, m_obs, c_syn, m_syn)

            if h_syn is None or np.sum(h_syn) <= (np.sum(h_obs) * 2):
                return -np.inf
            else:
                # h_syn = h_syn / np.sum(h_syn)
                # epsilon = 1e-8  # 1e-20
                # epsilon = (1 / np.sum(h_syn)) * 1e-2
                # epsilon = 1e-2  # version2
                # h_syn = h_syn + epsilon
                # h_syn = h_syn / np.sum(h_syn)
                # h_syn = h_syn * (50000 / np.sum(h_syn)) # version2 归一化到50000
                # h_syn = h_syn + epsilon
                # h_syn = np.where(h_syn < epsilon, epsilon, h_syn)  # version2
                # h_syn = h_syn / np.mean(h_syn) # versiob2 为了使典型到log（p）=0
                # lnlike = np.sum(h_obs * np.log10(h_syn))
                # lnlike = np.sum(h_obs * np.log(h_syn))
                # ! 增加 H_obs 归一化，是否正确？这样的话，似然函数的大小不会受到 Nstar 的影响；也能减小似然的整体数值
                # ! 如果不把 H_obs 也归一化，lnlike负太大，导致 exp(-lnlike)=0, 是均为0！
                # ! 不能对 H_obs 做归一化，没道理！
                # h_obs = h_obs / np.sum(h_obs)

                #  Hsyn 归一化到 Hobs, 返回 nan
                # n_syn = len(sample_syn)  # 我发现了盲点！
                # n_obs = len(sample_obs)
                # h_syn = h_syn / (n_syn / n_obs)

                # lnlike = np.sum(h_obs * np.log(h_syn))
                # # * NOTE correction, make max(lnlike_list)=0 !! IN corner_tests.draw_corner.py !!
                # delta = np.max(lnlike)
                # lnlike = lnlike - delta

                # version1
                # epsilon = (1 / np.sum(h_syn)) * 1e-2   # 改进epsilon的值
                # h_syn = h_syn / np.sum(h_syn)
                # h_syn = np.where(h_syn < epsilon, epsilon, h_syn)
                # lnlike = np.sum(h_obs * np.log(h_syn))

                # version2
                epsilon = 1e-2
                h_syn = h_syn * (50000 / np.sum(h_syn))  # 归一化到50000
                h_syn = np.where(h_syn < epsilon, epsilon, h_syn)
                h_syn = h_syn / np.mean(h_syn)  # 为了使典型的log(p)=0
                lnlike = np.sum(h_obs * np.log(h_syn))

                return lnlike

            # h_syn = h_syn / np.sum(h_syn)
            # epsilon = 1e-20
            # h_syn = h_syn + epsilon
            # h_syn = h_syn / np.sum(h_syn)
            # # lnlike = np.sum(h_obs * np.log10(h_syn))
            # # lnlike = np.sum(h_obs * np.log(h_syn))
            # lnlike = np.sum(h_obs * np.log(h_syn))  # / (len(sample_obs) * 100.)
            # # # * NOTE correction, make max(lnlike_list)=0 !! IN corner_tests.draw_corner.py !!
            # # delta = np.max(lnlike)
            # # lnlike = lnlike - delta
            # return lnlike
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
                aux = np.sum(h_obs * np.log(h_syn))  # / (len(sample_obs) * 100.)
                lnlikes.append(aux)
            lnlike = np.sum(lnlikes)
            # # * NOTE correction, make max(lnlike_list)=0 !! IN corner_tests.draw_corner.py !!
            # delta = np.max(lnlike)
            # lnlike = lnlike - delta
            return lnlike

    def get_funcname(self):
        funcname = self.func
        return funcname


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


# @log_time
def lnlike_5p(theta, step, isoc, likelihoodfunc, synstars, n_stars, sample_obs, position, times=1):
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
    if position == 'MW':  # Gaia MW
        condition = ((logage > 10.0) or (logage < 6.7) or (mh < -2.) or (mh > 0.4) or
                     (dm < 3.) or (dm > 15.) or (Av < 0.) or (Av > 3.) or (fb < 0.0) or (fb > 1.))
    elif position == 'LG':  # CSST Local Group
        condition = ((logage > 10.0) or (logage < 6.7) or (mh < -2.) or (mh > 0.4) or
                     (dm < 15.) or (dm > 19.) or (Av < 0.) or (Av > 2.) or (fb < 0.0) or (fb > 1.))  # dm > 28.
    else:
        condition = False

    if condition:
        return -np.inf
    else:
        if times == 1:
            sample_syn, isoc_obs = synstars(theta, n_stars, isoc, logage_step=logage_step, mh_step=mh_step)
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


# @log_time
def lnlike(theta_args,
           step,
           isoc,
           likelihoodfunc,
           synstars,
           n_stars,
           sample_obs,
           position,
           times=1,
           **kwargs):
    """

    Parameters
    ----------
    theta_args
    step
    isoc
    likelihoodfunc
    synstars
    n_stars
    sample_obs
    position
    times
    kwargs : 'logage', 'mh', 'dm', 'Av', 'fb', 'alpha'


    Returns
    -------

    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    logage_step, mh_step = step
    if len(theta_args) == 2:  # (fb, Av)
        logage = kwargs.get('logage')
        mh = kwargs.get('mh')
        dm = kwargs.get('dm')
        Av = kwargs.get('Av')
        fb, alpha = theta_args

        logage = round_to_step(logage, logage_step)
        mh = round_to_step(mh, mh_step)
        theta = logage, mh, dm, Av, fb, alpha

    elif len(theta_args) == 3:  # (logage, DM, Av)
        mh = kwargs.get('mh')
        fb = kwargs.get('fb')
        alpha = kwargs.get('alpha')
        logage, dm, Av = theta_args

        logage = round_to_step(logage, logage_step)
        mh = round_to_step(mh, mh_step)
        theta = logage, mh, dm, Av, fb, alpha

    elif len(theta_args) == 6:
        logage, mh, dm, Av, fb, alpha = theta_args
        logage = round_to_step(logage, logage_step)
        mh = round_to_step(mh, mh_step)
        theta = logage, mh, dm, Av, fb, alpha

    # !NOTE: theta range, dm(LMC,SMC~18.5, M31~24.5)
    # !      Av(for M31) range [0, 3] Fouesneau2014(https://iopscience.iop.org/article/10.1088/0004-637X/786/2/117)
    # !                               Li Lu MIMO & PhD thesis
    # !      dm(for Gaia) range [3, 15] Li Lu MIMO & PhD thesis
    # * Note [M/H] range in [-2, 0.7]? Dias2021 from [-0.9, 0.7]
    if position == 'MW':  # Gaia MW
        condition = ((logage > 10.0) or (logage < 6.7) or (mh < -2.) or (mh > 0.4) or
                     (dm < 3.) or (dm > 15.) or (Av < 0.) or (Av > 3.) or (fb < 0.) or (fb > 1.) or
                     (alpha < 0.5) or (alpha > 5.0))
    elif position == 'LG':
        # CSST Local Group
        # M31:24.47(https://ui.adsabs.harvard.edu/abs/2005MNRAS.356..979M/abstract)
        # LMC:18.5
        condition = ((logage > 10.0) or (logage < 6.7) or (mh < -2.) or (mh > 0.4) or
                     (dm < 15.) or (dm > 22.) or (Av < 0.) or (Av > 2.) or (fb < 0.) or (fb > 1.) or
                     (alpha < 0.) or (alpha > 5.))  # dm > 28.
    else:
        condition = False

    if condition:
        return -np.inf
    else:
        if times == 1:
            if likelihoodfunc.get_funcname() == 'DeltaCMD':
                sample_syn = synstars.delta_color_samples(theta, n_stars, isoc, logage_step=logage_step,
                                                          mh_step=mh_step)
            else:
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
                if likelihoodfunc.get_funcname() == 'DeltaCMD':
                    sample_syn = synstars.delta_color_samples(theta, n_stars, isoc, logage_step=logage_step,
                                                              mh_step=mh_step)
                else:
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


def get_CMD(sample_obs, sample_syn, model, photsys):
    c_obs, m_obs = CMD.extract_cmd(sample_obs, model, photsys, False)
    c_syn, m_syn = CMD.extract_cmd(sample_syn, model, photsys, True)
    return c_obs, m_obs, c_syn, m_syn


def bin_fixed(sample_obs, sample_syn, model, photsys, bins):
    h_obs, xe_obs, ye_obs = CMD.extract_hist2d(
        False, sample_obs, model, photsys, bins
    )
    h_syn, _, _ = CMD.extract_hist2d(
        True, sample_syn, model, photsys, bins=(xe_obs, ye_obs)
    )
    return h_obs, h_syn


def bin_knuth(c_obs, m_obs, c_syn, m_syn):
    """
    NOT recommand!!
    Parameters
    ----------
    sample_obs
    sample_syn
    model
    photsys

    Returns
    -------

    """
    bin_edges = [knuth_bin_width(c_obs, return_bins=True, quiet=True)[1],
                 knuth_bin_width(m_obs, return_bins=True, quiet=True)[1]]
    h_obs, _, _ = np.histogram2d(c_obs, m_obs, bins=bin_edges)
    h_syn, _, _ = np.histogram2d(c_syn, m_syn, bins=bin_edges)
    return h_obs, h_syn


def bin_blocks(c_obs, m_obs, c_syn, m_syn):
    bin_edges = [bayesian_blocks(c_obs),
                 bayesian_blocks(m_obs)]
    h_obs, _, _ = np.histogram2d(c_obs, m_obs, bins=bin_edges)
    h_syn, _, _ = np.histogram2d(c_syn, m_syn, bins=bin_edges)
    return h_obs, h_syn
