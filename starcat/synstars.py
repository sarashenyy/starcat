import numpy as np
import pandas as pd

from . import config, IMF
from .isoc import Isoc
from .logger import log_time


class SynStars(object):
    imf: IMF

    def __init__(self, model, photsys, imf, n_stars, binmethod, photerr):
        """
        Synthetic cluster samples.

        Parameters
        ----------
        model : str
            'parsec' or 'MIST'
        photsys : str
            'gaiaDR2' or 'daiaEDR3'
        imf : starcat.IMF
        n_stars : int
        binmethod :
            subclass of starcat.BinMethod: BinMS(), BinSimple()
        photerr :
            subclass of starcat.Photerr: GaiaEDR3()
        """
        self.imf = imf
        self.n_stars = n_stars
        self.model = model
        self.photsys = photsys
        self.binmethod = binmethod
        self.photerr = photerr

        source = config.config[self.model][self.photsys]
        self.bands = source['bands']
        self.mag_max = source['mag_max']
        self.bands = source['bands']
        self.mini = source['mini']
        self.mag = source['mag']

    @log_time
    def __call__(self, theta, variable_type_isoc, *args, **kwargs):
        """
        Make synthetic cluster sample, considering binary method and photmetry error.
        Need to instantiate Isoc()(optional), sunclass of BinMethod and subclass of Photerr first.
        BinMethod : BinMS(), BinSimple()
        Photerr : GaiaEDR3()
        Isoc(Parsec()) / Isoc(MIST()) : optinal

        Parameters
        ----------
        theta : tuple
            logage, mh, dist, Av, fb
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
        logage, mh, dist, Av, fb = theta
        logage_step = kwargs.get('logage_step')
        mh_step = kwargs.get('mh_step')
        # !step 1: logage, mh ==> isoc [phase, mini, [bands]]
        if isinstance(variable_type_isoc, pd.DataFrame):
            isoc = variable_type_isoc
        elif isinstance(variable_type_isoc, Isoc):
            isoc = variable_type_isoc.get_isoc(
                self.photsys, logage=logage, mh=mh, logage_step=logage_step, mh_step=mh_step
            )
        else:
            print('Please input an variable_type_isoc of type pd.DataFrame or starcat.Isoc.')

        # !step 2: add distance and Av, make observed iso
        isoc_new = self.get_observe_isoc(isoc, dist, Av)

        # !step 3: sample isochrone with specified Binary Method
        #         ==> n_stars [ mass x [_pri, _sec], bands x [_pri, _sec, _syn]
        sample_syn = self.sample_stars(isoc_new, fb)

        # !step 4: add photometry error for synthetic sample
        sample_syn = self.photerr.add_syn_photerr(sample_syn)

        # !step 5: discard nan&inf values primarily due to failed interpolate
        columns_to_check = self.bands
        sample_syn = sample_syn.dropna(subset=columns_to_check, how='any').reset_index(drop=True)
        for column in columns_to_check:
            sample_syn = sample_syn[~np.isinf(sample_syn[column])]
        sample_syn = sample_syn.reset_index(drop=True)

        return sample_syn

    def define_mass(self, isoc):
        """

        Parameters
        ----------
        isoc : pd.DataFrame
        dm : float
        """
        mass_min = min(
            isoc[(isoc[self.mag]) <= self.mag_max][self.mini]
        )
        mass_max = max(isoc[self.mini])
        return mass_min, mass_max

    def sample_stars(self, isoc, fb):
        """
        Create sample of synthetic stars with specified binary method.

        Parameters
        ----------
        isoc : pd.DataFrame
        fb : float

        Returns
        -------
        pd.DataFrame :
            sample_syn ==> [ mass x [_pri, _sec], bands x [_pri, _sec, _syn]
        """
        # define mass range
        mass_min, mass_max = self.define_mass(isoc=isoc)
        # create synthetic sample of length n_stars
        sample_syn = pd.DataFrame(np.zeros((self.n_stars, 1)), columns=['mass_pri'])
        sample_syn['mass_pri'] = self.imf.sample(n_stars=self.n_stars, mass_min=mass_min, mass_max=mass_max)

        # using specified binary method, see detail in binary.py
        sample_syn = self.binmethod.add_binary(
            fb, self.n_stars, sample_syn, isoc, self.imf, self.model, self.photsys
        )
        return sample_syn

    def get_observe_isoc(self, isoc, dist, Av):
        columns = isoc.columns
        isoc_new = pd.DataFrame(columns=columns)
        col_notin_bands = list(set(columns) - set(self.bands))
        isoc_new[col_notin_bands] = isoc[col_notin_bands]
        for _ in self.bands:
            # get extinction coeficients
            l, w, c = ext_coefs(_)
            #    sample_syn[_] += dm
            isoc_new[_] = isoc[_] + 5. * np.log10(dist * 1.e3) - 5. + c * Av
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
