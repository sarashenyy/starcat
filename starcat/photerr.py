from abc import ABC, abstractmethod

import numpy as np

from . import config
from .calSN import calculateSkyPix, calculateSN
from .logger import log_time
from .magerr import MagError


class Photerr(ABC):
    @abstractmethod
    def add_syn_photerr(self, sample_syn, **kwargs):
        """

        Parameters
        ----------
        sample_syn : pd.DataFrame
            [ mass x [_pri, _sec], bands x [_pri, _sec], bands ]
        kwargs

        Returns
        -------
        pd.DataFrames
            [ mass x [_pri, _sec], bands x [_pri, _sec], bands, bands x [_err] ]

        """
        pass


class CSSTsim(Photerr):
    def __init__(self, model):
        """

        Parameters
        ----------
        model : str
            'parsec' or 'MIST'.
        """
        self.photsys = 'CSST'
        self.model = model

        source = config.config[self.model][self.photsys]
        self.bands = source['bands']

    @log_time
    def add_syn_photerr(self, sample_syn, ex_time=150, ex_num=1):
        """

        Parameters
        ----------
        sample_syn : pd.DataFrame
            [ mass x [_pri, _sec], bands x [_pri, _sec], bands ]
        ex_time : float
            exposure time, typical time for main survey is 150s. Default is 150
        ex_num : int
            the number of exposures. Default is 1

        Returns
        -------
        pd.DataFrames
            [ mass x [_pri, _sec], bands x [_pri, _sec], bands, bands x [_err] ]
        """
        num = len(sample_syn)
        for i in range(len(self.bands)):
            skyPix = calculateSkyPix(fil=self.bands[i], zodi_mag=21.0)
            mag_syn = np.array(sample_syn[self.bands[i]])
            sn = calculateSN(sky=skyPix, ex_num=ex_num, t=ex_time, fil=self.bands[i], ABmag=mag_syn)
            mag_err = 1 / sn * 2.5 / np.log(10)
            standard = np.random.normal(0, 1, size=num)
            sample_syn[self.bands[i]] = mag_syn + mag_err * standard
            sample_syn['%s_err' % self.bands[i]] = mag_err * standard
        return sample_syn


class GaiaEDR3(Photerr):
    def __init__(self, model, med_nobs):
        """

        Parameters
        ----------
        model : str
            'parsec' or 'MIST'.
        med_nobs : list[int]
            Median observation number of each band. [G, BP, RP]
        """
        self.photsys = 'gaiaEDR3'
        self.model = model

        source = config.config[self.model][self.photsys]
        self.bands = source['bands']
        self.med_nobs = med_nobs

    @log_time
    def add_syn_photerr(self, sample_syn, **kwargs):
        """
        Add synthetic photoerror to synthetic star sample.

        Parameters
        ----------
        sample_syn : pd.DataFrame, cotaining [bands] cols
            Synthetic sample without photoerror.
        **kwargs :
            -
        Returns
        -------
        pd.DataFrames
            [ mass x [_pri, _sec], bands x [_pri, _sec], bands, bands x [_err] ]
        """
        e = MagError(med_nobs=self.med_nobs, bands=self.bands)
        # test when med_err is needed
        # g_med_err, bp_med_err, rp_med_err = e.estimate_med_photoerr(sample_syn=sample_syn)
        # sample_syn[self.bands[0] + '_err_syn'], sample_syn[self.bands[1] + '_err_syn'], sample_syn[
        #     self.bands[2] + '_err_syn'] = (
        #     g_med_err, bp_med_err, rp_med_err)
        g_syn, bp_syn, rp_syn = e(sample_syn=sample_syn)
        g0, bp0, rp0 = (
            np.array(sample_syn[self.bands[0]]), np.array(sample_syn[self.bands[1]]),
            np.array(sample_syn[self.bands[2]])
        )
        g_err, bp_err, rp_err = g_syn - g0, bp_syn - bp0, rp_syn - rp0

        sample_syn[self.bands[0]], sample_syn[self.bands[1]], sample_syn[self.bands[2]] = (
            g_syn, bp_syn, rp_syn
        )
        (sample_syn['%s_err' % self.bands[0]], sample_syn['%s_err' % self.bands[1]],
         sample_syn['%s_err' % self.bands[2]]) = (
            g_err, bp_err, rp_err
        )
        return sample_syn
