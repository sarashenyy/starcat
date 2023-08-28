import os

import numpy as np
import pandas as pd

from .bspline import Edr3LogMagUncertainty

module_dir = os.path.dirname(__file__)
spline_csv = os.path.join(module_dir, 'data', 'LogErrVsMagSpline.csv')


class MagError(Edr3LogMagUncertainty):
    """
    Gaia Mag [G, BP, RP] -> Mag_err [G_err, BP_err, RP_err].
    Return median uncertainty for a sample which n_obs obeys poisson distribution (hypothesis)

    method1 : using Edr3LogMagUncertainty
    method2 : using observation mag_magerr relation (corrected Nobs effect)

    Attributes
    ----------
    bands : list[str]
        Photometric band columns of the synthetic sample.
    spline_g, spline_bp, spline_rp
        Bspline function of each band.
    med_nobs : list[int]
        Median observation number of each band.
    """

    def __init__(
            self,
            sample_obs: pd.DataFrame = None,
            med_nobs: list[int] = None,
            spline_param: str = None,
            bands: list[str] = None,
            nobs: list[str] = None) -> None:
        if spline_param is None:
            spline_param = spline_csv
        super(MagError, self).__init__(spline_param)
        if bands is None:
            bands = ['Gmag', 'G_BPmag', 'G_RPmag']
        self.bands = bands
        self.spline_g = self._Edr3LogMagUncertainty__splines['g']
        self.spline_bp = self._Edr3LogMagUncertainty__splines['bp']
        self.spline_rp = self._Edr3LogMagUncertainty__splines['rp']
        if med_nobs:
            self.med_nobs = med_nobs
        else:
            if sample_obs is not None:
                if nobs is not None:
                    self.med_nobs = MagError.extract_med_nobs(sample_obs, nobs)
                else:
                    self.med_nobs = MagError.extract_med_nobs(sample_obs)
            else:
                raise ValueError('please enter med_nobs OR sample_obs')

    @staticmethod
    def extract_med_nobs(sample_obs: pd.DataFrame, nobs: list[str] = None) -> list[int]:
        """
        Extract the median value of n_obs(number of observation).

        Returns
        -------
        med_nobs : list[int]
            Three median observation number of each band (Gband, BPband, RPband).
        """
        if nobs is None:
            nobs = ['phot_g_n_obs', 'phot_bp_n_obs', 'phot_rp_n_obs']
        med_nobs = []
        for i in range(3):
            med = int(np.median(sample_obs[nobs[i]]))
            med_nobs.append(med)
        return med_nobs

    def random_n_obs(self, n_stars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate n_obs(number of observation) which obeys poisson(miu=med_nobs).

        Returns
        -------
        g_n_obs, bp_n_obs, rp_n_obs : tuple[np.ndarray, np.ndarray, np.ndarray]
            Synthetic observation number of each synthetic star.
        """
        g_n_obs = np.random.poisson(self.med_nobs[0], n_stars)
        bp_n_obs = np.random.poisson(self.med_nobs[1], n_stars)
        rp_n_obs = np.random.poisson(self.med_nobs[2], n_stars)
        return g_n_obs, bp_n_obs, rp_n_obs

    def estimate_med_photoerr(self, sample_syn: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate the photometric error (standard error) using bspline provided by Gaia.

        Parameters
        ----------
        sample_syn : pd.DataFrame
            Synthetic stars without photometric error.

        Returns
        -------
        g_med_err, bp_med_err, rp_med_err : tuple[np.ndarray, np.ndarray, np.ndarray]
            Standard error of the photometric error distribution, considering observation number of each synthetic star.
        """
        # step 1 : generate synthetic n_obs for each band
        n_stars = len(sample_syn)
        # noinspection PyTypeChecker
        g_n_obs, bp_n_obs, rp_n_obs = self.random_n_obs(n_stars)
        # step 2 : calculate mag_err when Nobs = 200(for G) / 20(for BP,RP)
        # g_med_err = np.sqrt(
        #     ( 10**(self.spline_g(sample_syn[self.bands[0]]) - np.log10(np.sqrt(g_n_obs) / np.sqrt(200))) )**2
        #     + (0.0027553202)**2
        # )
        # bp_med_err = np.sqrt(
        #     ( 10**(self.spline_bp(sample_syn[self.bands[1]]) - np.log10(np.sqrt(bp_n_obs) / np.sqrt(20))) )**2
        #     + (0.0027901700)**2
        # )
        # rp_med_err = np.sqrt(
        #     ( 10**(self.spline_rp(sample_syn[self.bands[2]]) - np.log10(np.sqrt(rp_n_obs) / np.sqrt(20))) )**2
        #     + (0.0037793818)**2
        # )
        # catch error: moved in to magerr_test.py
        # try:
        #     with warnings.catch_warnings():
        #         warnings.filterwarnings("error")
        #         g_med_err = np.sqrt(
        #             (10 ** (self.spline_g(sample_syn[self.bands[0]]) - np.log10(
        #                 np.sqrt(g_n_obs) / np.sqrt(200))) / 0.67) ** 2
        #             + 0.0027553202 ** 2
        #         )
        #         bp_med_err = np.sqrt(
        #             (10 ** (self.spline_bp(sample_syn[self.bands[1]]) - np.log10(
        #                 np.sqrt(bp_n_obs) / np.sqrt(20))) / 0.67) ** 2
        #             + 0.0027901700 ** 2
        #         )
        #         rp_med_err = np.sqrt(
        #             (10 ** (self.spline_rp(sample_syn[self.bands[2]]) - np.log10(
        #                 np.sqrt(rp_n_obs) / np.sqrt(20))) / 0.67) ** 2
        #             + 0.0037793818 ** 2
        #         )
        # except RuntimeWarning as warning:
        #     print('Caught a RuntimeWarning:', warning)
        #     print('g_n_obs:', g_n_obs)
        #     print('sample_syn:', sample_syn)
        #     raise warning
        g_med_err = np.sqrt(
            (10 ** (self.spline_g(sample_syn[self.bands[0]]) - np.log10(
                np.sqrt(g_n_obs) / np.sqrt(200))) / 0.67) ** 2
            + 0.0027553202 ** 2
        )
        bp_med_err = np.sqrt(
            (10 ** (self.spline_bp(sample_syn[self.bands[1]]) - np.log10(
                np.sqrt(bp_n_obs) / np.sqrt(20))) / 0.67) ** 2
            + 0.0027901700 ** 2
        )
        rp_med_err = np.sqrt(
            (10 ** (self.spline_rp(sample_syn[self.bands[2]]) - np.log10(
                np.sqrt(rp_n_obs) / np.sqrt(20))) / 0.67) ** 2
            + 0.0037793818 ** 2
        )
        return g_med_err, bp_med_err, rp_med_err

    def syn_sample_photoerr(self, sample_syn: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Add synthetic photometric error to synthetic stars.
        Return synthetic band mag (with statistic error) which obey with N(band,band_med_err).

        Returns
        -------
        g_syn, bp_syn, rp_syn : tuple[np.ndarray, np.ndarray, np.ndarray]
            Synthetic magnitude with photometic error of each band.
        """
        n_stars = len(sample_syn)
        normal_sample = np.random.normal(size=n_stars)
        g_med_err, bp_med_err, rp_med_err = self.estimate_med_photoerr(sample_syn)
        # g_syn = (g_med_err/0.67) * normal_sample + sample_syn[self.bands[0]]
        # bp_syn = (bp_med_err/0.67) * normal_sample + sample_syn[self.bands[1]]
        # rp_syn = (rp_med_err/0.67) * normal_sample + sample_syn[self.bands[2]]
        g_syn = g_med_err * normal_sample + sample_syn[self.bands[0]]
        bp_syn = bp_med_err * normal_sample + sample_syn[self.bands[1]]
        rp_syn = rp_med_err * normal_sample + sample_syn[self.bands[2]]
        return g_syn, bp_syn, rp_syn
