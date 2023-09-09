from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from . import config, IMF
from .isoc import Isoc
from .magerr import MagError


class Photerr(ABC):
    @abstractmethod
    def add_syn_photerr(self, sample_syn, **kwargs):
        pass


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
        pd.DataFrame : [bands] x [_syn]
        """
        e: MagError = MagError(med_nobs=self.med_nobs, bands=self.bands)
        # g_med_err, bp_med_err, rp_med_err = e.estimate_med_photoerr(sample_syn=sample_syn)
        # sample_syn[self.bands[0] + '_err_syn'], sample_syn[self.bands[1] + '_err_syn'], sample_syn[
        #     self.bands[2] + '_err_syn'] = (
        #     g_med_err, bp_med_err, rp_med_err)
        g_syn, bp_syn, rp_syn = e(sample_syn=sample_syn)
        sample_syn[self.bands[0] + '_syn'], sample_syn[self.bands[1] + '_syn'], sample_syn[
            self.bands[2] + '_syn'] = (
            g_syn, bp_syn, rp_syn)
        return sample_syn


class SynStars(object):
    imf: IMF

    def __init__(self, model, photsyn, imf, n_stars, binmethod, photerr):
        """
        Synthetic cluster samples.

        Parameters
        ----------
        model : str
            'parsec' or 'MIST'
        photsyn : str
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
        self.photsyn = photsyn
        self.binmethod = binmethod
        self.photerr = photerr

        source = config.config[self.model][self.photsyn]
        self.bands = source['bands']
        self.mag_max = source['mag_max']
        self.bands = source['bands']
        self.mini = source['Mini']
        self.mag = source['mag']

    def __call__(self, theta, step, variable_type_isoc, *args, **kwargs):
        """
        Make synthetic cluster sample, considering binary method and photmetry error.
        Need to instantiate Isoc()(optional), sunclass of BinMethod and subclass of Photerr first.
        BinMethod : BinMS(), BinSimple()
        Photerr : GaiaEDR3()
        Isoc(Parsec()) / Isoc(MIST()) : optinal

        Parameters
        ----------
        theta : tuple
            logage, mh, fb, dm
        step : tuple
            logage_step, mh_step
        isoc :
            starcat.Isoc() or pd.DataFRame, optional
            - starcat.Isoc() : Isoc(Parsec()), Isoc(MIST()). Default is None.
            - pd.DataFrame : isoc [phase, mini, [bands]]
        """
        logage, mh, fb, dm = theta
        logage_step, mh_step = step

        # step 1: logage, mh ==> isoc [phase, mini, [bands]]
        if isinstance(variable_type_isoc, pd.DataFrame):
            isoc = variable_type_isoc
        elif isinstance(variable_type_isoc, Isoc):
            isoc = variable_type_isoc.get_isoc(
                self.photsyn, logage=logage, mh=mh, logage_step=logage_step, mh_step=mh_step
            )
        else:
            print('Please input an variable_type_isoc of type pd.DataFrame or starcat.Isoc.')

        # step 2: sample isochrone with specified Binary Method
        #         ==> n_stars [ mass x [_pri, _sec], bands x [_pri, _sec, _syn]
        sample_syn = self.sample_stars(isoc, fb, dm)

        # step 3: add distance module
        for _ in self.bands:
            sample_syn[_ + '_syn'] += dm

        # step 4: add photometry error for synthetic sample
        sample_syn = self.photerr.add_syn_photerr(sample_syn)

        return sample_syn

    def define_mass(self, isoc, dm):
        """

        Parameters
        ----------
        isoc : pd.DataFrame
        dm : float
        """
        mass_min = min(
            isoc[(isoc[self.mag] + dm) <= self.mag_max][self.mini]
        )
        mass_max = max(isoc[self.mini])
        return mass_min, mass_max

    def sample_stars(self, isoc, fb, dm):
        """
        Create sample of synthetic stars with specified binary method.

        Parameters
        ----------
        isoc : pd.DataFrame
        fb : float
        dm : float

        Returns
        -------
        pd.DataFrame :
            sample_syn ==> [ mass x [_pri, _sec], bands x [_pri, _sec, _syn]
        """
        # define mass range
        mass_min, mass_max = self.define_mass(isoc=isoc, dm=dm)
        # create synthetic sample of length n_stars
        sample_syn = pd.DataFrame(np.zeros((self.n_stars, 1)), columns=['mass_pri'])
        sample_syn['mass_pri'] = self.imf.sample(n_stars=self.n_stars, mass_min=mass_min, mass_max=mass_max)

        # using specified binary method, see detail in binary.py
        sample_syn = self.binmethod.add_binary(
            fb, self.n_stars, sample_syn, isoc, self.imf, self.model, self.photsyn, dm
        )
        return sample_syn
