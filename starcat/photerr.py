from abc import ABC, abstractmethod

import numpy as np

from . import config
from .calSN import calculateSkyPix, calculateSN
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

    # @log_time
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
            mag_err = (1 / sn) * (2.5 / np.log(10))  # ln(10)

            standard = np.random.normal(0, 1, size=num)

            # max_err = 2   # if standard > 2 * scale
            # condition = np.abs(standard) > max_err
            # indices = np.where(condition)[0]
            # new_standard = np.random.normal(0, 1, size=len(indices))
            # standard[indices] = new_standard

            # lower, upper = -2.0, 2.0
            # mu, sigma = 0, 1
            # standard = stats.truncnorm.rvs(
            #     (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=num)

            sample_syn[self.bands[i]] = mag_syn + mag_err * standard
            sample_syn['%s_err' % self.bands[i]] = mag_err * standard
        return sample_syn


class GaiaDR3(Photerr):
    def __init__(self,
                 model,
                 med_nobs=(200., 20., 20.)):
        """

        Parameters
        ----------
        model : str
            'parsec' or 'MIST'.
        med_nobs : list[int]
            Median observation number of each band. [G, BP, RP]
        """
        self.photsys = 'gaiaDR3'
        self.model = model

        source = config.config[self.model][self.photsys]
        self.bands = source['bands']
        self.med_nobs = med_nobs

    # @log_time
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

    def get_error(self, sample_syn):
        e = MagError(med_nobs=self.med_nobs, bands=self.bands)
        g_med_err, bp_med_err, rp_med_err = e.estimat_emed_photoerr(sample_syn)
        return g_med_err, bp_med_err, rp_med_err


class Individual(Photerr):
    def __init__(self, model, photsys, observation, add_error='none'):
        """

        Parameters
        ----------
        model
        photsys
        observation : pd.DataFrame
            observation data
        add_error : string
            'absolute error' or 'relative error' or 'none'
        """
        # model details
        source = config.config[model][photsys]
        self.bands = source['bands']

        # observation details
        o_source = config.config['observation'][photsys]
        self.o_bands = o_source['bands']
        self.oe_bands = o_source['bands_err']
        # self.prob = o_source['Prob']

        # get mag-magerr relation(data points) from observation
        self.mag_magerr = []
        # o_sample = observation[observation[self.prob]>0.7]
        o_sample = observation
        for i in range(len(self.o_bands)):
            aux = o_sample.loc[:, [self.o_bands[i], self.oe_bands[i]]]
            aux = aux.sort_values(by=self.o_bands[i], ascending=True)
            self.mag_magerr.append(aux)

        self.add_error = add_error

    def add_syn_photerr(self, sample_syn, **kwargs):
        """

        Parameters
        ----------
        sample_syn
        *kwargs :
            absolute_error : float

        Returns
        -------

        """
        num = len(sample_syn)
        for j in range(len(self.bands)):
            xp = self.mag_magerr[j][self.o_bands[j]]
            fp = self.mag_magerr[j][self.oe_bands[j]]
            mag_syn = np.array(sample_syn[self.bands[j]])
            mag_err = np.interp(mag_syn, xp, fp)

            # test for spread uncertainty
            if self.add_error == 'absolute error':
                absolute_err = kwargs.get('absolute_error')
                if self.bands[j] == 'G':
                    mag_err = np.sqrt(np.square(mag_err) + np.square(absolute_err))
                elif self.bands[j] == 'BP' or self.bands[j] == 'RP':
                    mag_err = np.sqrt(np.square(mag_err) + np.square(absolute_err / np.sqrt(2)))

            elif self.add_error == 'relative error':
                relative_err = 0.001  # delta as parameter
                absolute_err = relative_err * mag_syn
                mag_err = np.sqrt(np.square(mag_err) + np.square(absolute_err))

            standard = np.random.normal(0, 1, size=num)
            sample_syn[self.bands[j]] = mag_syn + mag_err * standard
            sample_syn['%s_err' % self.bands[j]] = mag_err * standard
        return sample_syn
