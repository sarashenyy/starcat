import numpy as np
import pandas as pd

from . import config, IMF
from .isoc import Isoc
from .logger import log_time


class SynStars(object):
    imf: IMF

    def __init__(self, model, photsys, imf, binmethod, photerr):
        """
        Synthetic cluster samples.

        Parameters
        ----------
        model : str
            'parsec' or 'MIST'
        photsys : str
            'gaiaDR2' or 'daiaEDR3'
        imf : starcat.IMF
        binmethod :
            subclass of starcat.BinMethod: BinMS(), BinSimple()
        photerr :
            subclass of starcat.Photerr: GaiaEDR3()
        """
        self.imf = imf
        # self.n_stars = n_stars
        self.model = model
        self.photsys = photsys
        self.binmethod = binmethod
        self.photerr = photerr

        source = config.config[self.model][self.photsys]
        self.bands = source['bands']
        self.band_max_obs = source['band_max']
        self.band_max_syn = [x + 0.5 for x in source['band_max']]
        self.bands = source['bands']
        self.mini = source['mini']
        # self.mag = source['mag']

    @log_time
    def __call__(self, theta, n_stars, variable_type_isoc, test=False, **kwargs):
        """
        Make synthetic cluster sample, considering binary method and photmetry error.
        Need to instantiate Isoc()(optional), sunclass of BinMethod and subclass of Photerr first.
        BinMethod : BinMS(), BinSimple()
        Photerr : GaiaEDR3()
        Isoc(Parsec()) / Isoc(MIST()) : optinal

        Parameters
        ----------
        theta : tuple
            logage, mh, dm, Av, fb
        n_stars : int
        step : tuple
            logage_step, mh_step
        isoc :
            starcat.Isoc() or pd.DataFRame, optional
            - starcat.Isoc() : Isoc(Parsec()), Isoc(MIST()). Default is None.
            - pd.DataFrame : isoc [phase, mini, [bands]]
        *kwargs :
            logage_step
            mh_step
        """
        logage, mh, dm, Av, fb = theta
        logage_step = kwargs.get('logage_step')
        mh_step = kwargs.get('mh_step')
        # !step 1: logage, mh ==> isoc [phase, mini, [bands]]
        if isinstance(variable_type_isoc, pd.DataFrame):
            isoc = variable_type_isoc
        elif isinstance(variable_type_isoc, Isoc):
            isoc = variable_type_isoc.get_isoc(
                self.photsys, logage=logage, mh=mh, logage_step=logage_step, mh_step=mh_step
            )
            if isoc is False:  # get_isoc() raise Error
                return False
        else:
            print('Please input an variable_type_isoc of type pd.DataFrame or starcat.Isoc.')

        # !step 2: add distance modulus and Av, make observed iso
        isoc_new = self.get_observe_isoc(isoc, dm, Av)

        # ?inspired by batch rejection sampling
        samples = pd.DataFrame()
        accepted = 0
        # batch_size = int(n_stars * 10)
        # runtime test
        batch_size = int(n_stars)
        test_sample_time = 0

        while accepted < n_stars:
            # !step 3: sample isochrone with specified Binary Method
            #         ==> n_stars [ mass x [_pri, _sec], bands x [_pri, _sec, _syn]
            sample_syn = self.sample_stars(isoc_new, batch_size, fb)

            # !step 4: add photometry error for synthetic sample
            sample_syn = self.photerr.add_syn_photerr(sample_syn)

            # !step 5: (deleted!) 1. discard nan&inf values primarily due to failed interpolate
            # !        2. 只有在所有波段都暗于极限星等时才丢弃！即只有当一颗星在所有波段都不可见时才丢弃，只要这颗星在任一波段可见，则保留。
            # !           意味着最终返回的 samples[band] 中包含暗于该波段极限星等的值
            # # 1. discard nan&inf values primarily due to failed interpolate
            # columns_to_check = self.bands
            # sample_syn = sample_syn.dropna(subset=columns_to_check, how='any').reset_index(drop=True)
            # for column in columns_to_check:
            #     sample_syn = sample_syn[~np.isinf(sample_syn[column])]
            # # 2. discard sample_syn[mag] > band_max
            # if len(self.mag) == 1:
            #     sample_syn = sample_syn[sample_syn[self.mag[0]] <= self.band_max_obs[0]]
            # elif len(self.mag) > 1:
            #     for mag_col, band_max_val in zip(self.mag, self.band_max_obs):
            #         sample_syn = sample_syn[sample_syn[mag_col] <= band_max_val]

            # condition为所有波段都暗于极限星等的星，将之丢弃
            condition = sample_syn[self.bands[0]] > self.band_max_obs[0]
            for b, b_max in zip(self.bands[1:], self.band_max_obs[1:]):
                cond = sample_syn[b] > b_max
                condition = condition & cond

            sample_syn = sample_syn[~condition].reset_index(drop=True)
            samples = pd.concat([samples, sample_syn], ignore_index=True)
            accepted += len(sample_syn)
            # dynamically adjusting rejection rate
            # rejection_rate = 1 - len(sample_syn) / batch_size
            # if rejection_rate > 0.2:
            #     batch_size = int(batch_size * 1.2)
            # else:
            #     batch_size = int(batch_size * 0.8)

            # runtime test
            test_sample_time += 1

        samples = samples.iloc[:n_stars]
        # return samples
        # runtime test
        accepted_rate = accepted / (batch_size * test_sample_time)
        if test is True:
            return samples, accepted_rate, test_sample_time
        else:
            return samples

    def define_mass(self, isoc):
        """

        Parameters
        ----------
        isoc : pd.DataFrame
        dm : float

        """
        mass_max = max(isoc[self.mini])
        # if len(self.mag) == 1:
        #     try:
        #         mass_min = min(
        #             isoc[(isoc[self.mag[0]]) <= self.band_max_syn[0]][self.mini]
        #         )
        #     except ValueError:
        #         print('Do not have any stars brighter than the mag range!!')

        # elif len(self.mag) > 1:
        #     mass_min = min(
        #         isoc[(isoc[self.mag[0]]) <= self.band_max_syn[0]][self.mini]
        #     )
        #     for i in range(len(self.mag)):
        #         aux_min = min(
        #             isoc[(isoc[self.mag[i]]) <= self.band_max_syn[i]][self.mini]
        #         )
        #         if aux_min < mass_min:
        #             mass_min = aux_min

        aux_list = []
        for i in range(len(self.bands)):
            # synthetic Mini range is slightly larger than the observed for the consideration of binary and photerror
            condition = isoc[self.bands[i]] <= self.band_max_syn[i]
            filtered_isoc = isoc[condition]

            if not filtered_isoc.empty:
                aux_min = min(filtered_isoc[self.mini])
                aux_list.append(aux_min)
        mass_min = min(aux_list)

        return mass_min, mass_max

    def sample_stars(self, isoc, n_stars, fb):
        """
        Create sample of synthetic stars with specified binary method.

        Parameters
        ----------
        isoc : pd.DataFrame
        n_stars : int
        fb : float

        Returns
        -------
        pd.DataFrame :
            sample_syn ==> [ mass x [_pri, _sec], bands x [_pri, _sec, _syn]
        """
        # define mass range
        mass_min, mass_max = self.define_mass(isoc=isoc)
        # create synthetic sample of length n_stars
        sample_syn = pd.DataFrame(np.zeros((n_stars, 1)), columns=['mass_pri'])
        sample_syn['mass_pri'] = self.imf.sample(n_stars=n_stars, mass_min=mass_min, mass_max=mass_max)

        # using specified binary method, see detail in binary.py
        sample_syn = self.binmethod.add_binary(
            fb, n_stars, sample_syn, isoc, self.imf, self.model, self.photsys
        )
        return sample_syn

    def get_observe_isoc(self, isoc, dm, Av):
        columns = isoc.columns
        isoc_new = pd.DataFrame(columns=columns)
        col_notin_bands = list(set(columns) - set(self.bands))
        isoc_new[col_notin_bands] = isoc[col_notin_bands]
        for _ in self.bands:
            # get extinction coeficients
            l, w, c = ext_coefs(_)
            #    sample_syn[_] += dm
            isoc_new[_] = isoc[_] + dm + c * Av
        return isoc_new


def ext_coefs(band):
    """
    From PARSEC CMD

    Parameters
    ----------
    band: list

    Returns
    -------
    λeff (Å), ωeff (Å), Aλ/AV
    """
    sys_param = {'NUV': [2887.74, 609, 1.88462],
                 'u': [3610.40, 759, 1.55299],
                 'g': [4811.96, 1357, 1.19715],
                 'r': [6185.81, 1435, 0.86630],
                 'i': [7641.61, 1536, 0.66204],
                 'z': [9043.96, 1108, 0.47508],
                 'y': [9660.53, 633, 0.42710]}

    return sys_param[band]
