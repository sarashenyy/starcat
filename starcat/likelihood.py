import math
import warnings
from abc import ABC, abstractmethod
from math import comb

import numpy as np
import pandas as pd
from astropy.stats import knuth_bin_width, bayesian_blocks
from robustgp import ITGP
from scipy import signal
from scipy.stats import energy_distance, gaussian_kde

from .cmd import CMD
from .widgets import round_to_step


class LikelihoodFunc(ABC):
    @abstractmethod
    def eval_lnlike(self, sample_syn):
        pass

    def get_funcname(self):
        pass


class Gaussian2D(LikelihoodFunc):
    """
    lnlike(H_{syn},H_{obs}) = -\frac{1}{2}\sum{\frac{(H_{obs}-H_{syn})^2}{H_{obs}+H_{syn}+1}}
    """

    def __init__(self,
                 model,
                 photsys,
                 bin_method,
                 number=1,
                 bins=None,
                 **kwargs):
        """

        Parameters
        ----------
        model
        photsys
        bins : array-like, optional, [float, float]
            [color_bin_width, mag_bin_width]
        number : int
            number of CMDs
        kwargs :
            'sample_obs'
        """
        if bins is None:
            bins = [0.2, 0.5]

        self.func = 'Hist2Hoint'
        self.model = model
        self.photsys = photsys
        self.number = number

        self.sample_obs = kwargs.get('sample_obs')
        self.c_obs, self.m_obs = CMD.extract_cmd(self.sample_obs, self.model, self.photsys, False)
        # self.cobs_err, self.mobs_err =  CMD.extract_error(self.sample_obs, self.model, self.photsys, False)

        self.bin_method = bin_method
        if self.bin_method == 'fixed':
            self.bins = bins
            c_bw, m_bw = self.bins
            c_bins = np.arange(start=np.min(self.c_obs) - 0.5 * c_bw, stop=np.max(self.c_obs) + 0.5 * c_bw, step=c_bw)
            m_bins = np.arange(start=np.min(self.m_obs) - 0.5 * m_bw, stop=np.max(self.m_obs) + 0.5 * m_bw, step=m_bw)
            self.bin_edges = [c_bins, m_bins]

        elif self.bin_method == 'knuth':
            c_bw, c_bins = knuth_bin_width(self.c_obs, return_bins=True, quiet=True)
            m_bw, m_bins = knuth_bin_width(self.m_obs, return_bins=True, quiet=True)
            self.bins = [c_bw, m_bw]
            self.bin_edges = [c_bins, m_bins]

        elif self.bin_method == 'blocks':
            self.bins_edges = [bayesian_blocks(self.c_obs),
                               bayesian_blocks(self.m_obs)]

        elif self.bin_method == 'halfknuth':
            c_bw, _ = knuth_bin_width(self.c_obs, return_bins=True, quiet=True)
            m_bw, _ = knuth_bin_width(self.m_obs, return_bins=True, quiet=True)
            self.bins = [c_bw / 2., m_bw / 2.]
            c_bw, m_bw = self.bins
            c_bins = np.arange(start=np.min(self.c_obs) - 0.5 * c_bw, stop=np.max(self.c_obs) + 0.5 * c_bw, step=c_bw)
            m_bins = np.arange(start=np.min(self.m_obs) - 0.5 * m_bw, stop=np.max(self.m_obs) + 0.5 * m_bw, step=m_bw)
            self.bin_edges = [c_bins, m_bins]

        self.h_obs, _, _ = np.histogram2d(self.c_obs, self.m_obs, bins=self.bin_edges)

    # @log_time
    def eval_lnlike(self, sample_syn):
        if self.number == 1:
            h_syn = None

            c_syn, m_syn = CMD.extract_cmd(sample_syn, self.model, self.photsys, True)
            h_syn, _, _ = np.histogram2d(c_syn, m_syn, bins=self.bin_edges)

            if h_syn is None or np.sum(h_syn) <= (np.sum(self.h_obs) * 5):
                return -np.inf
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
            else:
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

                # lnlike = -(0.5 / n_obs) * np.sum(np.square(h_obs - h_syn) / (h_obs + h_syn + 1))
                # # * NOTE correction, make max(lnlike_list)=0 !! IN corner_tests.draw_corner.py !!
                # delta = np.max(lnlike)
                # lnlike = lnlike - delta

                # epsilon = 1e-2
                # h_syn = np.where(h_syn < ((1/n_syn) * epsilon), (1/n_syn) * epsilon, h_syn)
                # h_obs = np.where(h_obs < ((1/n_obs) * epsilon), (1/n_obs) * epsilon, h_obs)
                #
                # lnlike = np.sum(
                #     - 0.5 * np.log(2 * np.pi * (h_obs + h_syn))
                #     - 0.5 * (h_obs - h_syn) ** 2 / (h_obs + h_syn)
                # )

                # version 1
                n_syn = np.sum(h_syn)
                n_obs = np.sum(self.h_obs)
                h_syn = n_obs * h_syn / n_syn
                # epsilon = 1.0
                # lnlike = -0.5 * np.sum(np.square(self.h_obs - h_syn) / (self.h_obs + h_syn + epsilon))
                # version 1.5
                epsilon = 1 / self.h_obs.size
                lnlike = -0.5 * np.sum(np.square(self.h_obs - h_syn) / (self.h_obs + h_syn + epsilon))

                # version 2
                # n_syn = np.sum(h_syn)
                # n_obs = np.sum(self.h_obs)
                # h_syn = h_syn / n_syn
                # h_obs = self.h_obs / n_obs
                # lnlike = -0.5 * np.sum(np.square(h_obs - h_syn))

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
    """
    only for binary fraction and alpha
    """

    def __init__(self,
                 model,
                 photsys,
                 bin_method,
                 bins=None,
                 **kwargs):
        """
        dCMD : delta color(obs - rigdeline) vs. mag

        Parameters
        ----------
        model
        photsys
        bin_method : string
        bins : [float, float] / [int, int], optional
            [int, int] : binnum_dc, binnum_m
            [float, float] : binwidth_dc, binwidth_m
        kwargs :
            'sample_obs'
        """

        self.func = 'DeltaCMD'
        self.model = model
        self.photsys = photsys

        if bins is None:
            bins = [21, 30]

        # if isinstance(bins[0], int):
        #     self.binnum_dc, self.binnum_w = bins[0], bins[1]
        # elif isinstance(bins[0], float):
        #     self.bw_dc, self.bw_m = bins[0], bins[1]

        self.sample_obs = kwargs.get('sample_obs')
        self.c_obs, self.m_obs = CMD.extract_cmd(self.sample_obs, self.model, self.photsys, False)

        # find ridgeline, calculate delta color
        self.ridgeline = find_rigdeline(self.c_obs, self.m_obs)
        self.rc_obs, _ = self.ridgeline.predict(self.m_obs.reshape(-1, 1))
        self.rc_obs = self.rc_obs.ravel()
        self.dc_obs = self.c_obs - self.rc_obs

        self.bin_method = bin_method
        if self.bin_method == 'fixed':
            self.bins = bins
            if isinstance(bins[0], int):
                dc_binnum, m_binnum = self.bins
                dc_bins = np.linspace(start=np.min(self.dc_obs), stop=np.max(self.dc_obs), num=dc_binnum + 1)
                m_bins = np.linspace(start=np.min(self.m_obs), stop=np.max(self.m_obs), num=m_binnum + 1)
            elif isinstance(bins[0], float):
                dc_bw, m_bw = self.bins
                dc_bins = np.arange(start=np.min(self.dc_obs) - 0.5 * dc_bw, stop=np.max(self.m_obs) + 0.5 * dc_bw,
                                    step=dc_bw)
                m_bins = np.arange(start=np.min(self.m_obs) - 0.5 * m_bw, stop=np.max(self.m_obs) + 0.5 * m_bw,
                                   step=m_bw)
            self.bin_edges = [dc_bins, m_bins]

        elif self.bin_method == 'knuth':
            dc_bw, dc_bins = knuth_bin_width(self.dc_obs, return_bins=True, quiet=True)
            m_bw, m_bins = knuth_bin_width(self.m_obs, return_bins=True, quiet=True)
            self.bin_edges = [dc_bins, m_bins]

        self.dh_obs, _, _ = np.histogram2d(self.dc_obs, self.m_obs, bins=self.bin_edges)

    def eval_lnlike(self, sample_syn, sample_obs=None):
        dh_syn = None
        c_syn, m_syn = CMD.extract_cmd(sample_syn, self.model, self.photsys, True)
        rc_syn, _ = self.ridgeline.predict(m_syn.reshape(-1, 1))
        rc_syn = rc_syn.ravel()
        dc_syn = c_syn - rc_syn
        dh_syn, _, _ = np.histogram2d(dc_syn, m_syn, bins=self.bin_edges)

        if dh_syn is None or np.sum(dh_syn) <= (np.sum(self.dh_obs) * 5):
            return -np.inf
        else:
            epsilon = (1 / np.sum(dh_syn) * 1e-2)
            dh_syn = dh_syn / np.sum(dh_syn)
            dh_syn = np.where(dh_syn < epsilon, epsilon, dh_syn)
            dh_obs_e = np.where(self.dh_obs < 1.0, 1e-2, self.dh_obs)
            lnlike = np.sum(
                (self.dh_obs * np.log(dh_syn)) - (self.dh_obs * np.log(dh_obs_e))
            )

        return lnlike
        # source = config.config[self.model][self.photsys]
        # m_obs = sample_obs[source['mag']].values.ravel()
        # dc_obs = sample_obs['delta_c'].values.ravel()
        # m_syn = sample_syn[source['mag']].values.ravel()
        # dc_syn = sample_syn['delta_c'].values.ravel()
        #
        # h_syn = None
        # h_obs = None
        # if self.bin_method == 'knuth':
        #     h_obs, h_syn = bin_knuth(dc_obs, m_obs, dc_syn, m_syn)
        #
        # elif self.bin_method == 'blocks':
        #     h_obs, h_syn = bin_blocks(dc_obs, m_obs, dc_syn, m_syn)
        #
        # if h_syn is None or np.sum(h_syn) == 0:
        #     return -np.inf
        # else:
        #     # h_syn = h_syn / np.sum(h_syn)
        #     epsilon = 1e-20
        #     h_syn = h_syn + epsilon
        #     h_syn = h_syn / np.sum(h_syn)
        #     h_obs = h_obs / np.sum(h_obs)
        #
        #     lnlike = np.sum(h_obs * np.log(h_syn))
        #     # delta = np.max(lnlike)
        #     # lnlike = lnlike - delta
        #     return lnlike

    def get_funcname(self):
        funcname = self.func
        return funcname


def find_rigdeline(color, mag):
    res = ITGP(mag, color,
               alpha1=0.5, alpha2=0.975, nsh=2, ncc=2, nrw=1,
               optimize_kwargs=dict(optimizer='lbfgsb')
               )
    gp, consistency = res.gp, res.consistency
    return gp


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


class KernelSmooth(LikelihoodFunc):
    def __init__(self,
                 model,
                 photsys,
                 # bin_method,
                 **kwargs):
        self.func = 'kernel'
        self.model = model
        self.photsys = photsys

    def eval_lnlike(self, sample_obs, sample_syn):
        pass
        # c_obs, m_obs, c_syn, m_syn = get_CMD(sample_obs, sample_syn, self.model, self.photsys)
        # cobs_err, mobs_err = CMD.extract_error(sample_obs, self.model, self.photsys, False)
        #
        # # bin_edges 按照观测最小的sigam_c和sigma_m（的1/2）为最小格子分bin
        # grid_min_c = np.min(cobs_err) / 2.
        # grid_min_m = np.min(mobs_err) / 2.
        # # [start, stop), with spacing between values given by step.
        # c_start = c_obs.min() - 3. * cobs_err[c_obs.argmin()]
        # c_end = c_obs.max() + 3. * cobs_err[c_obs.argmax()]
        # m_start = m_obs.min() - 3. * mobs_err[m_obs.argmin()]
        # m_end = m_obs.max() + 3. * mobs_err[m_obs.argmax()]
        #
        # c_bin = np.arange(start=c_start, stop=c_end + grid_min_c, step=grid_min_c)
        # m_bin = np.arange(start=m_start, stop=m_end + grid_min_m, step=grid_min_m)
        # bin_edges = [c_bin, m_bin]
        # # 按照 bin_edges 得到 H_syn, H_obs; 此时H_syn和H_obs应该拥有相同的格点
        # # https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
        # # np.histogram2d(x, y, bins=[x_edges, y_edges]) return ndarray, shape(nx, ny)
        # # H_obs, _, _ = np.histogram2d(c_obs, m_obs, bins=bin_edges)
        # H_syn, _, _ = np.histogram2d(c_syn, m_syn, bins=bin_edges)
        #
        # # 根据观测的(c,m)以及对应的(sigma_c,sigma_m)制作高斯mask:
        # # 1、获得观测的每颗星的格点索引（mask中心值索引）
        # maskid_center_color = np.digitize(c_obs, c_bin)  # DEFAULT right=False
        # maskid_center_mag = np.digitize(m_obs, m_bin)
        # # 2、根据 cobs_err 和 mobs_err 计算mask的形状 (切片左闭右开)
        # #    mask形状：在 color 轴上索引为 [maskid_center_color - halflen_color : maskid_center_color + halflen_color + 1]
        # #             在 mag 轴上索引为 [maskid_center_mag - halflen_mag : maskid_center_mag + halflen_mag + 1]
        # halflen_color = int((3. * cobs_err) / grid_min_c)
        # halflen_mag = int((3. * mobs_err) / grid_min_m)
        # # 3、生成高斯mask
        # for
        #
        # # 1、记录每个mask在整个大的直方图中的下标位置loc，
        # # 2、用gaussian_2d给mask矩阵赋值
        # # for each mask:
        # # 根据每个mask的下标位置loc，取出H_syn中与mask形状相同的矩阵syns
        #
        # pi = []
        # lnpi = []
        # for cobs, mobs, cobs_err, mobs_err in zip(c_obs, m_obs, cobs_err, mobs_err):
        #     aux_p = np.dot(syn.ravel(), mask.ravel())
        #     aux_lnp = np.log(aux_p)
        #     pi.append(aux_p)
        #     lnpi.append(aux_lnp)
        #
        # lnlike = np.sum(lnpi)
        # return lnlike


def generate_gaussian_mask(shape, sigma_x, sigma_y=None):
    """
    中间速度：34.6 µs ± 333 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

    Parameters
    ----------
    shape: (rows, cols) (x,y)
    sigma_x: sigma_rows
    sigma_y: sigma_cols

    Returns
    -------

    """
    if sigma_y is None:
        sigma_y = sigma_x
    rows, cols = shape

    def get_gaussian_fct(size, sigma):
        fct_gaus_x = np.linspace(0, size, size)
        fct_gaus_x = fct_gaus_x - size / 2
        fct_gaus_x = fct_gaus_x ** 2
        fct_gaus_x = fct_gaus_x / (2 * sigma ** 2)
        fct_gaus_x = np.exp(-fct_gaus_x)
        return fct_gaus_x

    mask = np.outer(get_gaussian_fct(rows, sigma_y), get_gaussian_fct(cols, sigma_x))
    return mask / mask.sum()


def gaussian_kernel(n, std, normalised=True):
    """
    最快：15.7 µs ± 103 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    Generates a n x n matrix with a centered gaussian
    of standard deviation std centered on it. If normalised,
    its volume equals 1.
    Note to make sure volume equals ~0.975 in n x n martix, std <= n x 1/5
    """
    gaussian1D = signal.windows.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= (2 * np.pi * (std ** 2))
    return gaussian2D


class Hist2Point4CMD(LikelihoodFunc):
    """
    * Non-normalized Cross Entropy
    likelihood = \prod{p_{i}^{n_i}}
    log(likelihood) =\sum{n_{i}\ln{p_{i}}}
    """

    def __init__(self,
                 model,
                 photsys,
                 bin_method,
                 number=1,
                 bins=None,
                 **kwargs):
        """

        Parameters
        ----------
        model
        photsys
        bins : array-like, optional, [float, float]
            [color_bin_width, mag_bin_width]
        number : int
            number of CMDs
        kwargs :
            'sample_obs'
        """
        if bins is None:
            bins = [0.2, 0.5]

        self.func = 'Hist2Point'
        self.model = model
        self.photsys = photsys
        self.number = number

        self.sample_obs = kwargs.get('sample_obs')
        self.c_obs, self.m_obs = CMD.extract_cmd(self.sample_obs, self.model, self.photsys, False)
        # self.cobs_err, self.mobs_err =  CMD.extract_error(self.sample_obs, self.model, self.photsys, False)

        self.bin_method = bin_method
        if self.bin_method == 'fixed':
            self.bins = bins
            c_bw, m_bw = self.bins
            c_bins = np.arange(start=np.min(self.c_obs) - 0.5 * c_bw, stop=np.max(self.c_obs) + 0.5 * c_bw, step=c_bw)
            m_bins = np.arange(start=np.min(self.m_obs) - 0.5 * m_bw, stop=np.max(self.m_obs) + 0.5 * m_bw, step=m_bw)
            self.bin_edges = [c_bins, m_bins]

        elif self.bin_method == 'knuth':
            c_bw, c_bins = knuth_bin_width(self.c_obs, return_bins=True, quiet=True)
            m_bw, m_bins = knuth_bin_width(self.m_obs, return_bins=True, quiet=True)
            self.bins = [c_bw, m_bw]
            self.bin_edges = [c_bins, m_bins]

        elif self.bin_method == 'blocks':
            self.bins_edges = [bayesian_blocks(self.c_obs),
                               bayesian_blocks(self.m_obs)]

        elif self.bin_method == 'halfknuth':
            c_bw, _ = knuth_bin_width(self.c_obs, return_bins=True, quiet=True)
            m_bw, _ = knuth_bin_width(self.m_obs, return_bins=True, quiet=True)
            self.bins = [c_bw / 2., m_bw / 2.]
            c_bw, m_bw = self.bins
            c_bins = np.arange(start=np.min(self.c_obs) - 0.5 * c_bw, stop=np.max(self.c_obs) + 0.5 * c_bw, step=c_bw)
            m_bins = np.arange(start=np.min(self.m_obs) - 0.5 * m_bw, stop=np.max(self.m_obs) + 0.5 * m_bw, step=m_bw)
            self.bin_edges = [c_bins, m_bins]

        self.h_obs, _, _ = np.histogram2d(self.c_obs, self.m_obs, bins=self.bin_edges)

    # @log_time
    def eval_lnlike(self, sample_syn):
        if self.number == 1:
            h_syn = None

            c_syn, m_syn = CMD.extract_cmd(sample_syn, self.model, self.photsys, True)
            h_syn, _, _ = np.histogram2d(c_syn, m_syn, bins=self.bin_edges)

            if h_syn is None or np.sum(h_syn) <= (np.sum(self.h_obs) * 5):
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
                epsilon = (1 / np.sum(h_syn)) * 1e-2  # 改进epsilon的值
                h_syn = h_syn / np.sum(h_syn)  # norm h_syn
                h_syn = np.where(h_syn < epsilon, epsilon, h_syn)
                lnlike = np.sum(self.h_obs * np.log(h_syn))

                # version1 norm h_obs
                # epsilon = (1 / np.sum(h_syn)) * 1e-2  # 改进epsilon的值
                # h_syn = h_syn / np.sum(h_syn)
                # h_syn = np.where(h_syn < epsilon, epsilon, h_syn)
                # h_obs = self.h_obs / np.sum(self.h_obs)  # norm h_obs
                # lnlike = np.sum(h_obs * np.log(h_syn))

                # version2
                # epsilon = 1e-2  # epsilon = 1e-2, for Melotte_22, epsilon=1e-1 is not a solution
                # h_syn = h_syn * (50000 / np.sum(h_syn))  # 归一化到50000
                # h_syn = np.where(h_syn < epsilon, epsilon, h_syn)
                # h_syn = h_syn / np.mean(h_syn)  # 为了使典型的log(p)=0, np.mean(h_syn) = np.sum(h_syn)/bins
                # lnlike = np.sum(self.h_obs * np.log(h_syn))

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
        # elif self.number > 1:
        #     source = config.config[self.model][self.photsys]
        #     epsilon = 1e-20
        #     lnlikes = []
        #     for i in range(self.number):
        #         m_obs = sample_obs[source['mag'][i]]
        #         c_obs = sample_obs[source['color'][i][0]] - sample_obs[source['color'][i][1]]
        #         m_syn = sample_syn[source['mag'][i]]
        #         c_syn = sample_syn[source['color'][i][0]] - sample_syn[source['color'][i][1]]
        #         if isinstance(self.bins, int):
        #             hist_obs = plt.hist2d(c_obs, m_obs, self.bins)
        #             h_obs, xe_obs, ye_obs = hist_obs[0], hist_obs[1], hist_obs[2]
        #             hist_syn = plt.hist2d(c_syn, m_syn, self.bins)
        #             h_syn, xe_syn, ye_syn = hist_syn[0], hist_syn[1], hist_syn[2]
        #         elif isinstance(self.bins, tuple):
        #             h_obs, xe_obs, ye_obs = np.histogram2d(c_obs, m_obs, bins=self.bins)
        #             h_syn, xe_syn, ye_syn = np.histogram2d(c_syn, m_syn, bins=self.bins)
        #
        #         h_syn = h_syn / np.sum(h_syn)
        #         h_syn = h_syn + epsilon
        #         h_syn = h_syn / np.sum(h_syn)
        #         aux = np.sum(h_obs * np.log(h_syn))  # / (len(sample_obs) * 100.)
        #         lnlikes.append(aux)
        #     lnlike = np.sum(lnlikes)
        #     # # * NOTE correction, make max(lnlike_list)=0 !! IN corner_tests.draw_corner.py !!
        #     # delta = np.max(lnlike)
        #     # lnlike = lnlike - delta
        #     return lnlike

    def get_funcname(self):
        funcname = self.func
        return funcname


class Gaussian1D(LikelihoodFunc):
    """
    * WRONG!!
    likelihood = \prod{p_{i}^{n_i}}
    log(likelihood) =\sum{n_{i}\ln{p_{i}}}
    """

    def __init__(self,
                 model,
                 photsys,
                 bin_method,
                 number=1,
                 bins=None,
                 **kwargs):
        """

        Parameters
        ----------
        model
        photsys
        bins : array-like, optional, [float, float]
            [color_bin_width, mag_bin_width]
        number : int
            number of CMDs
        kwargs :
            'sample_obs'
        """
        if bins is None:
            bins = [0.2, 0.5]

        self.func = 'Gaussian'
        self.model = model
        self.photsys = photsys
        self.number = number

        self.sample_obs = kwargs.get('sample_obs')
        self.c_obs, self.m_obs = CMD.extract_cmd(self.sample_obs, self.model, self.photsys, False)
        # self.cobs_err, self.mobs_err =  CMD.extract_error(self.sample_obs, self.model, self.photsys, False)

        self.bin_method = bin_method
        if self.bin_method == 'fixed':
            self.bins = bins
            c_bw, m_bw = self.bins
            c_bins = np.arange(start=np.min(self.c_obs) - 0.5 * c_bw, stop=np.max(self.c_obs) + 0.5 * c_bw, step=c_bw)
            m_bins = np.arange(start=np.min(self.m_obs) - 0.5 * m_bw, stop=np.max(self.m_obs) + 0.5 * m_bw, step=m_bw)
            self.bin_edges = [c_bins, m_bins]

        elif self.bin_method == 'knuth':
            c_bw, c_bins = knuth_bin_width(self.c_obs, return_bins=True, quiet=True)
            m_bw, m_bins = knuth_bin_width(self.m_obs, return_bins=True, quiet=True)
            self.bins = [c_bw, m_bw]
            self.bin_edges = [c_bins, m_bins]

        elif self.bin_method == 'blocks':
            self.bins_edges = [bayesian_blocks(self.c_obs),
                               bayesian_blocks(self.m_obs)]

        elif self.bin_method == 'halfknuth':
            c_bw, _ = knuth_bin_width(self.c_obs, return_bins=True, quiet=True)
            m_bw, _ = knuth_bin_width(self.m_obs, return_bins=True, quiet=True)
            self.bins = [c_bw / 2., m_bw / 2.]
            c_bw, m_bw = self.bins
            c_bins = np.arange(start=np.min(self.c_obs) - 0.5 * c_bw, stop=np.max(self.c_obs) + 0.5 * c_bw, step=c_bw)
            m_bins = np.arange(start=np.min(self.m_obs) - 0.5 * m_bw, stop=np.max(self.m_obs) + 0.5 * m_bw, step=m_bw)
            self.bin_edges = [c_bins, m_bins]

        self.hc_obs, _ = np.histogram(self.c_obs, bins=self.bin_edges[0])
        self.hm_obs, _ = np.histogram(self.m_obs, bins=self.bin_edges[1])

    # @log_time
    def eval_lnlike(self, sample_syn):
        if self.number == 1:
            hc_syn = None
            hm_syn = None

            c_syn, m_syn = CMD.extract_cmd(sample_syn, self.model, self.photsys, True)
            hc_syn, _ = np.histogram(c_syn, bins=self.bin_edges[0])
            hm_syn, _ = np.histogram(m_syn, bins=self.bin_edges[1])

            nc_syn = np.sum(hc_syn)
            nm_syn = np.sum(hm_syn)
            nc_obs = np.sum(self.hc_obs)
            nm_obs = np.sum(self.hm_obs)

            if nc_syn == 0 or nm_syn == 0 or nc_syn <= 5 * nc_obs or nm_syn <= 5 * nm_obs:
                return -np.inf
            else:
                # normalize
                hc_syn = hc_syn / nc_syn
                hm_syn = hm_syn / nm_syn
                hc_obs = self.hc_obs / nc_obs
                hm_obs = self.hm_obs / nm_obs

                epsilon = 1e-2
                hc_syn = np.where(hc_syn < ((1 / nc_syn) * epsilon), (1 / nc_syn) * epsilon, hc_syn)
                hm_syn = np.where(hm_syn < ((1 / nm_syn) * epsilon), (1 / nm_syn) * epsilon, hm_syn)
                hc_obs = np.where(hc_obs < ((1 / nc_obs) * epsilon), (1 / nc_obs) * epsilon, hc_obs)
                hm_obs = np.where(hm_obs < ((1 / nm_obs) * epsilon), (1 / nm_obs) * epsilon, hm_obs)

                ln_c = (- 0.5 * np.log(2 * np.pi * (hc_obs + hc_syn))
                        - 0.5 * (hc_obs - hc_syn) ** 2 / (hc_obs + hc_syn))
                ln_m = (- 0.5 * np.log(2 * np.pi * (hm_obs + hm_syn))
                        - 0.5 * (hm_obs - hm_syn) ** 2 / (hm_obs + hm_syn))
                lnlike = np.sum(ln_c) + np.sum(ln_m)
                return lnlike

    def get_funcname(self):
        funcname = self.func
        return funcname


class KLD(LikelihoodFunc):
    """
    likelihood = \prod{p_{i}^{n_i}}
    log(likelihood) =\sum{n_{i}\ln{p_{i}}}
    """

    def __init__(self,
                 model,
                 photsys,
                 bin_method,
                 bins=None,
                 **kwargs):
        """

        Parameters
        ----------
        model
        photsys
        bins : array-like, optional, [float, float]
            [color_bin_width, mag_bin_width]
        number : int
            number of CMDs
        kwargs :
            'sample_obs'
        """
        if bins is None:
            bins = [0.2, 0.5]

        self.func = 'KLDivergence'
        self.model = model
        self.photsys = photsys

        self.sample_obs = kwargs.get('sample_obs')
        self.c_obs, self.m_obs = CMD.extract_cmd(self.sample_obs, self.model, self.photsys, False)
        # self.cobs_err, self.mobs_err =  CMD.extract_error(self.sample_obs, self.model, self.photsys, False)

        self.bin_method = bin_method
        if self.bin_method == 'fixed':
            self.bins = bins
            c_bw, m_bw = self.bins
            c_bins = np.arange(start=np.min(self.c_obs) - 0.5 * c_bw, stop=np.max(self.c_obs) + 0.5 * c_bw, step=c_bw)
            m_bins = np.arange(start=np.min(self.m_obs) - 0.5 * m_bw, stop=np.max(self.m_obs) + 0.5 * m_bw, step=m_bw)
            self.bin_edges = [c_bins, m_bins]

        elif self.bin_method == 'knuth':
            c_bw, c_bins = knuth_bin_width(self.c_obs, return_bins=True, quiet=True)
            m_bw, m_bins = knuth_bin_width(self.m_obs, return_bins=True, quiet=True)
            self.bins = [c_bw, m_bw]
            self.bin_edges = [c_bins, m_bins]

        elif self.bin_method == 'blocks':
            self.bins_edges = [bayesian_blocks(self.c_obs),
                               bayesian_blocks(self.m_obs)]

        elif self.bin_method == 'halfknuth':
            c_bw, _ = knuth_bin_width(self.c_obs, return_bins=True, quiet=True)
            m_bw, _ = knuth_bin_width(self.m_obs, return_bins=True, quiet=True)
            self.bins = [c_bw / 2., m_bw / 2.]
            c_bw, m_bw = self.bins
            c_bins = np.arange(start=np.min(self.c_obs) - 0.5 * c_bw, stop=np.max(self.c_obs) + 0.5 * c_bw, step=c_bw)
            m_bins = np.arange(start=np.min(self.m_obs) - 0.5 * m_bw, stop=np.max(self.m_obs) + 0.5 * m_bw, step=m_bw)
            self.bin_edges = [c_bins, m_bins]

        elif self.bin_method == 'halfequal':
            # euqal frequency for mag axis
            binnum_m = int(len(self.m_obs) ** (2 / 5) * 2)
            _, m_bins = pd.qcut(self.m_obs, q=binnum_m, retbins=True, labels=False, duplicates='drop')
            c_bw = 0.2
            c_bins = np.arange(start=np.min(self.c_obs) - 0.5 * c_bw, stop=np.max(self.c_obs) + 0.5 * c_bw, step=c_bw)
            self.bin_edges = [c_bins, m_bins]

        self.h_obs, _, _ = np.histogram2d(self.c_obs, self.m_obs, bins=self.bin_edges)

    # @log_time
    def eval_lnlike(self, sample_syn, sample_obs=None):
        """
        NOTE : self.h_obs and h_syn both sum to 1.

        Parameters
        ----------
        sample_syn
        sample_obs

        Returns
        -------

        """
        h_syn = None
        c_syn, m_syn = CMD.extract_cmd(sample_syn, self.model, self.photsys, True)
        h_syn, _, _ = np.histogram2d(c_syn, m_syn, bins=self.bin_edges)

        if h_syn is None or np.sum(h_syn) <= (np.sum(self.h_obs) * 5):
            return -np.inf

        else:
            # version1
            # epsilon = (1 / np.sum(h_syn)) * 1e-2  # 改进epsilon的值
            # h_syn = h_syn / np.sum(h_syn)
            # h_syn = np.where(h_syn < epsilon, epsilon, h_syn)
            # lnlike = np.sum(h_obs * np.log(h_syn))

            # version2
            # epsilon = 1e-2  # epsilon = 1e-2, for Melotte_22, epsilon=1e-1 is not a solution
            # h_syn = h_syn * (50000 / np.sum(h_syn))  # 归一化到50000
            # h_syn = np.where(h_syn < epsilon, epsilon, h_syn)
            # h_syn = h_syn / np.mean(h_syn)  # 为了使典型的log(p)=0, np.mean(h_syn) = np.sum(h_syn)/bins
            # lnlike = np.sum(h_obs * np.log(h_syn))

            epsilon = (1 / np.sum(h_syn)) * 1e-2  # 改进epsilon的值
            h_syn = h_syn / np.sum(h_syn)  # norm h_syn
            h_syn = np.where(h_syn < epsilon, epsilon, h_syn)
            h_obs_e = np.where(self.h_obs < 1.0, 1e-2, self.h_obs)  # 为后续np.log()计算而处理0值，不改变h_obs本身
            lnlike = np.sum(
                (self.h_obs * np.log(h_syn)) - (self.h_obs * np.log(h_obs_e))
            )

            return lnlike

    def get_funcname(self):
        funcname = self.func
        return funcname


def kl_div(p, q):
    """
    Kullback-Leibler divergence D(P || Q) for discrete distributions
    Epsilon is used here to avoid infinity result
    """
    epsilon = 1e-8
    # p = p + epsilon
    q = q + epsilon

    # divergence = np.sum(p * np.log(p / q))
    divergence = np.sum(np.where(p != 0.0, p * np.log(p / q), 0.0))
    return divergence


# def lnlike_2p(theta_age_mh, fb, dm, step, isoc, likelihoodfunc, synstars, n_stars, sample_obs):
#     """
#
#     Parameters
#     ----------
#     synstars : starcat.SynStars
#         the instantiated SynStars()
#     likelihoodfunc : subclass
#         subclass of LikelihoodFunc : Hist2Hist(), Hist2Point()
#     sample_obs : pd.DataFrame
#     isoc : starcat.Isoc()
#         the instantiated Isoc().
#         ```python
#         p = Parsec()
#         i = Isoc(p)
#         ```
#     theta_age_mh : tuple
#         logage, mh
#     fb : float
#     dm : float
#     step : tuples
#         logage_step, mh_step
#
#     Returns
#     -------
#
#     """
#     logage, mh = theta_age_mh
#     theta = logage, mh, fb, dm
#     lnlike = lnlike_5p(theta, step, isoc, likelihoodfunc, synstars, n_stars, sample_obs)
#     return lnlike


# @log_time
# def lnlike_5p(theta, step, isoc, likelihoodfunc, synstars, n_stars, sample_obs, position, times=1):
#     # try:
#     warnings.filterwarnings("ignore", category=RuntimeWarning)
#     logage, mh, dm, Av, fb = theta
#     logage_step, mh_step = step
#     logage = round_to_step(logage, logage_step)
#     mh = round_to_step(mh, mh_step)
#     theta = logage, mh, dm, Av, fb
#     # !NOTE: theta range, dm(LMC,SMC~18.5, M31~24.5)
#     # !      Av(for M31) range [0, 3] Fouesneau2014(https://iopscience.iop.org/article/10.1088/0004-637X/786/2/117)
#     # !                               Li Lu MIMO & PhD thesis
#     # !      dm(for Gaia) range [3, 15] Li Lu MIMO & PhD thesis
#     # * Note [M/H] range in [-2, 0.7]? Dias2021 from [-0.9, 0.7]
#     if position == 'MW':  # Gaia MW
#         condition = ((logage > 10.0) or (logage < 6.7) or (mh < -2.) or (mh > 0.4) or
#                      (dm < 3.) or (dm > 15.) or (Av < 0.) or (Av > 3.) or (fb < 0.0) or (fb > 1.))
#     elif position == 'LG':  # CSST Local Group
#         condition = ((logage > 10.0) or (logage < 6.7) or (mh < -2.) or (mh > 0.4) or
#                      (dm < 15.) or (dm > 19.) or (Av < 0.) or (Av > 2.) or (fb < 0.0) or (fb > 1.))  # dm > 28.
#     else:
#         condition = False
#
#     if condition:
#         return -np.inf
#     else:
#         if times == 1:
#             sample_syn, isoc_obs = synstars(theta, n_stars, isoc, logage_step=logage_step, mh_step=mh_step)
#             if sample_syn is False:
#                 # return 1e10
#                 return -np.inf
#             else:
#                 lnlike = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
#                 return lnlike
#             # lnlike = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
#             # return lnlike
#
#         elif times > 1:
#             # * without acceleration
#             lnlike_list = []
#             for i in range(times):
#                 sample_syn = synstars(theta, n_stars, isoc, logage_step=logage_step, mh_step=mh_step)
#                 # if sample_syn is False:
#                 #     lnlike_one = 1e10
#                 # else:
#                 #     lnlike_one = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
#                 lnlike_one = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
#                 lnlike_list.append(lnlike_one)
#             lnlike = np.sum(lnlike_list) / times
#
#             # * acceleration with parallelization
#             # /home/shenyueyue/Packages/miniconda3/envs/mcmc/lib/python3.10/site-packages/joblib/externals/loky/backend
#             # /resource_tracker.py:318: UserWarning: resource_tracker:
#             # There appear to be 278 leaked semlock objects to clean up at shutdown
#             #   warnings.warn('resource_tracker: There appear to be %d '
#
#             # def compute_lnlike_one_iteration(i):
#             #     sample_syn = synstars(theta, isoc, logage_step=logage_step, mh_step=mh_step)
#             #     lnlike_one = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
#             #     return lnlike_one
#             #
#             # lnlike_list = Parallel(n_jobs=-1)(delayed(compute_lnlike_one_iteration)(i) for i in range(times))
#             # # lnlike_list = Parallel(n_jobs=-1, temp_folder='/home/shenyueyue/Projects/starcat/temp_folder')(
#             # #     delayed(compute_lnlike_one_iteration)(i) for i in range(times))
#             # lnlike = np.sum(lnlike_list) / times
#
#             return lnlike
#     # except ZeroDivisionError as e:
#     #     # Handle the division by zero error here
#     #     print(f"!!!!ZeroDivisionError occurred: {e}")
#     #     print(theta)
#     #     # You can add custom handling or return a specific value as needed
#     #     return None  # Or any value that makes sense in your context
#     # import warnings
#     # warnings.filterwarnings("ignore", category=RuntimeWarning)
#     # try:
#     #     sample_syn = synstars(theta, step, isoc)
#     #     lnlike = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
#     #     return lnlike
#     # except RuntimeWarning:
#     #     print(theta)
#     #     return -np.inf


# @log_time
def lnlike(theta_args,
           step,
           isoc,
           likelihoodfunc,
           synstars,
           n_stars,
           # sample_obs,
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

    elif len(theta_args) == 7:
        logage, mh, dm, Av, fb, alpha, err = theta_args
        logage = round_to_step(logage, logage_step)
        mh = round_to_step(mh, mh_step)
        theta = logage, mh, dm, Av, fb, alpha, err

    # !NOTE: theta range, dm(LMC,SMC~18.5, M31~24.5)
    # !      Av(for M31) range [0, 3] Fouesneau2014(https://iopscience.iop.org/article/10.1088/0004-637X/786/2/117)
    # !                               Li Lu MIMO & PhD thesis
    # !      dm(for Gaia) range [3, 15] Li Lu MIMO & PhD thesis
    # * Note [M/H] range in [-2, 0.7]? Dias2021 from [-0.9, 0.7]
    if isinstance(alpha, list):
        alpha1, alpha2 = alpha
        condition_alpha = ((alpha1 < 0.) or (alpha1 > 5.0) or (alpha2 < 0.5) or (alpha2 > 5.0))
    elif isinstance(alpha, float):
        condition_alpha = ((alpha < 0.5) or (alpha > 5.0))

    if position == 'MW':  # Gaia MW
        condition_dm = ((dm < 3.) or (dm > 15.))

    elif position == 'LG':
        # CSST Local Group
        # M31:24.47(https://ui.adsabs.harvard.edu/abs/2005MNRAS.356..979M/abstract)
        # LMC:18.5
        condition_dm = (dm < 15.) or (dm > 22.)

    condition = ((logage > 10.0) or (logage < 6.7) or (mh < -2.) or (mh > 0.4) or
                 condition_dm or (Av < 0.) or (Av > 3.) or (fb < 0.) or (fb > 1.) or
                 condition_alpha)
    if condition:
        return -np.inf

    else:
        if times == 1:
            # if likelihoodfunc.get_funcname() == 'DeltaCMD':
            #     sample_syn = synstars.delta_color_samples(theta, n_stars, isoc, logage_step=logage_step,
            #                                               mh_step=mh_step)
            # else:
            #     sample_syn = synstars(theta, n_stars, isoc, logage_step=logage_step, mh_step=mh_step)
            sample_syn = synstars(theta, n_stars, isoc, logage_step=logage_step, mh_step=mh_step)

            if sample_syn is False:
                # return 1e10
                return -np.inf
            else:
                lnlike = likelihoodfunc.eval_lnlike(sample_syn)
                return lnlike
            # lnlike = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
            # return lnlike

        elif times > 1:
            # * without acceleration
            lnlike_list = []
            for i in range(times):
                # if likelihoodfunc.get_funcname() == 'DeltaCMD':
                #     sample_syn = synstars.delta_color_samples(theta, n_stars, isoc, logage_step=logage_step,
                #                                               mh_step=mh_step)
                # else:
                #     sample_syn = synstars(theta, n_stars, isoc, logage_step=logage_step, mh_step=mh_step)
                sample_syn = synstars(theta, n_stars, isoc, logage_step=logage_step, mh_step=mh_step)

                # if sample_syn is False:
                #     lnlike_one = 1e10
                # else:
                #     lnlike_one = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)

                # lnlike_one = likelihoodfunc.eval_lnlike(sample_obs, sample_syn)
                lnlike_one = likelihoodfunc.eval_lnlike(sample_syn)
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


def bin_knuth(
        c_obs,
        m_obs,
        c_syn,
        m_syn,
        usesyn=False):
    """
    usesyn=True, is NOT recommand!! For one observation CMD, different bins will be used,
    which may cause likelihood unstable.

    Parameters
    ----------
    c_obs
    m_obs
    c_syn
    m_syn
    usesyn

    Returns
    -------

    """
    if usesyn:

        # Define the ranges
        c_min, c_max = c_obs.min(), c_obs.max()
        m_min, m_max = m_obs.min(), m_obs.max()

        # Filter out data points outside the ranges
        mask = (c_syn >= c_min) & (c_syn <= c_max) & (m_syn >= m_min) & (m_syn <= m_max)
        c_syn_filtered = c_syn[mask]
        m_syn_filtered = m_syn[mask]

        bin_edges = [knuth_bin_width(c_syn_filtered, return_bins=True, quiet=True)[1],
                     knuth_bin_width(m_syn_filtered, return_bins=True, quiet=True)[1]]
    else:
        bin_edges = [knuth_bin_width(c_obs, return_bins=True, quiet=True)[1],
                     knuth_bin_width(m_obs, return_bins=True, quiet=True)[1]]

    h_obs, _, _ = np.histogram2d(c_obs, m_obs, bins=bin_edges)
    h_syn, _, _ = np.histogram2d(c_syn, m_syn, bins=bin_edges)
    return h_obs, h_syn


def bin_blocks(
        c_obs,
        m_obs,
        c_syn,
        m_syn,
        usesyn=True):
    """
    usesyn=True, is NOT recommand!! For one observation CMD, different bins will be used,
    which may cause likelihood unstable.

    Parameters
    ----------
    c_obs
    m_obs
    c_syn
    m_syn
    usesyn

    Returns
    -------

    """
    if usesyn:
        # Define the ranges
        c_min, c_max = c_obs.min(), c_obs.max()
        m_min, m_max = m_obs.min(), m_obs.max()

        # Filter out data points outside the ranges
        mask = (c_syn >= c_min) & (c_syn <= c_max) & (m_syn >= m_min) & (m_syn <= m_max)
        c_syn_filtered = c_syn[mask]
        m_syn_filtered = m_syn[mask]

        bin_edges = [bayesian_blocks(c_syn_filtered),
                     bayesian_blocks(m_syn_filtered)]
    else:
        bin_edges = [bayesian_blocks(c_obs),
                     bayesian_blocks(m_obs)]
    h_obs, _, _ = np.histogram2d(c_obs, m_obs, bins=bin_edges)
    h_syn, _, _ = np.histogram2d(c_syn, m_syn, bins=bin_edges)
    return h_obs, h_syn
