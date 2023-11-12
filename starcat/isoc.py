import os
from abc import ABC, abstractmethod

import joblib
import numpy as np
import pandas as pd
from berliner import CMD
from joblib import Parallel, delayed

from . import config
from .logger import log_time
from .widgets import round_to_step


class IsocModel(ABC):
    """
    An abstract base class for isochrone model that defines the interface for its subclass.
    """

    @abstractmethod
    def get_isoc(self, photsyn, **kwargs):
        """
        An abstarct method that must be implemented by subclasses to get isochrone from different model.

        Parameters
        ----------
        photsyn : str, optinal
            The synthetic photometry to use for isochrone. For example: "gaiaDR2". See config.toml for more options.
        kwargs : dict
            - logage (float): logarithmic age
            - logage_step (float): step size of logage
            - mh (float): [M/H]
            - mh_step (float): step size of [M/H]

        Returns
        -------
        pd.DataFrame: isochrone containing the evolutionary phase, initial mass and photometry bands.
        """
        pass

    def bulk_load(self, photsyn, n_jobs, **kwargs):
        """
        Bulk Laod isochrones.

        Parameters
        ----------
        n_jobs
        photsyn
        kwargs : dict
            - logage_grid (tuple): start, end, step
            - mh_grid (tuple): start, end, step
        """
        pass


class Parsec(IsocModel):
    """
    subclass for abstract base class Model()
    """

    def __init__(self):
        self.model = 'parsec'

    @log_time
    def get_isoc(self, photsyn, **kwargs):
        """
        Get isochrone from parsec model.
        Parameters
        ----------
        photsyn : str, optinal
            The synthetic photometry to use for isochrone. For example: "gaiaDR2". See config.toml for more options.
        **kwargs : dict
            - logage (float): logarithmic age
            - logage_step (float): step size of logage
            - mh (float): [M/H]
            - mh_step (float): step size of [M/H]

        Returns
        -------
        pd.DataFrame: isochrone containing the evolutionary phase, initial mass and photometry bands.
        """
        # logage = round_to_step(kwargs.get('logage'), step=kwargs.get('logage_step'))
        # mh = round_to_step(kwargs.get('mh'), step=kwargs.get('mh_step'))
        logage_step = kwargs.get('logage_step')
        mh_step = kwargs.get('mh_step')
        if logage_step is None or logage_step == 0:
            logage = kwargs.get('logage')
        else:
            logage = round_to_step(kwargs.get('logage'), step=kwargs.get('logage_step'))
        if mh_step is None or mh_step == 0:
            mh = kwargs.get('mh')
        else:
            mh = round_to_step(kwargs.get('mh'), step=kwargs.get('mh_step'))

        source = config.config[self.model][photsyn]
        bands_isoc = source['bands_isoc']
        bands = source['bands']
        mini = source['mini']
        mass = source['mass']
        label = source['label']
        phase = source['phase']
        isoc_dir = source['isoc_dir']
        isoc_path = config.data_dir + isoc_dir + f'age{logage:+.2f}_mh{mh:+.2f}.joblib'

        EOFE_flag = True
        if os.path.exists(isoc_path):
            try:
                isochrone = joblib.load(isoc_path)
                # change bands name
                flag = False
                for band_isoc, band in zip(bands_isoc, bands):
                    if band_isoc in isochrone.columns:
                        flag = True
                        isochrone = isochrone.rename(columns={band_isoc: band})
                if flag:
                    joblib.dump(isochrone, isoc_path)
            except EOFError:
                EOFE_flag = False

        elif os.path.exists(isoc_path) is False or EOFE_flag is False:
            isochrone = pd.DataFrame([])
            max_attempt = 5
            attempt_time = 0
            while isochrone.empty and attempt_time <= max_attempt:
                c = CMD()
                isochrone = c.get_one_isochrone(
                    logage=logage, z=None, mh=mh, photsys_file=photsyn
                )
                # truncate isochrone, PMS~EAGB
                # ATTENTION! parsec use "label" to represent evolutionary phase, different from MIST("phase")
                isochrone = isochrone[
                    (isochrone['label'] >= min(label)) & (isochrone['label'] <= max(label))].to_pandas()
                # add evolutionary phase info
                for i, element in enumerate(label):
                    index = np.where(isochrone['label'] == element)[0]
                    isochrone.loc[index, 'phase'] = phase[i]
                # change bands name
                rename_dict = dict(zip(bands_isoc, bands))
                isochrone = isochrone.rename(columns=rename_dict)
                # save isochrone file
                # if not os.path.exists(isoc_path):
                joblib.dump(isochrone, isoc_path)
                attempt_time += 1

        useful_columns = ['phase', mini, mass] + bands
        try:
            isoc = isochrone[useful_columns]
            return isoc
        except UnboundLocalError:
            print(f'logage={logage}, [M/H]={mh} occurs UnboundLocal Error in getting isochrone.')
            return False


    def bulk_load(self, photsyn, n_jobs, **kwargs):
        """
        Bulk Laod isochrones.

        Parameters
        ----------
        n_jobs
        photsyn
        kwargs : dict
            - logage_grid (tuple): start, end, step
            - mh_grid (tuple): start, end, step
        """
        astart, aend, astep = kwargs.get('logage_grid')
        mstart, mend, mstep = kwargs.get('mh_grid')
        abin = np.arange(astart, aend, astep)
        mbin = np.arange(mstart, mend, mstep)
        logage_mh = []
        for a in abin:
            for m in mbin:
                logage_mh.append([a, m])

        # nested function, access variable in parent function
        def bulk_load_wrapper(logage, mh):
            self.get_isoc(photsyn=photsyn, logage=logage, mh=mh, logage_step=astep, mh_step=mstep)

        # parallel excution
        Parallel(n_jobs=n_jobs)(
            delayed(bulk_load_wrapper)(logage, mh) for logage, mh in logage_mh
        )


class MIST(IsocModel):
    """
    subclass for abstract base class Model()
    """

    def __init__(self):
        self.model = 'mist'

    def get_isoc(self, photsyn, **kwargs):
        pass

    def bulk_load(self, photsyn, n_jobs, **kwargs):
        """
        Bulk Laod isochrones.

        Parameters
        ----------
        n_jobs
        photsyn
        kwargs : dict
            - logage_grid (tuple): start, end, step
            - mh_grid (tuple): start, end, step
        """
        pass


class Isoc(object):
    """
    Isochrone

    """

    def __init__(self, model):
        self.model = model

    def get_isoc(self, photsyn, **kwargs):
        """
        Get isochrone from model.

        Parameters
        ----------
        photsyn : str, optinal
            The synthetic photometry to use for isochrone. For example: "gaiaDR2". See config.toml for more options.
        kwargs : dict
            - logage (float): logarithmic age
            - logage_step (float): step size of logage
            - mh (float): [M/H]
            - mh_step (float): step size of [M/H]

        Returns
        -------
        pd.DataFrame: isochrone containing the evolutionary phase, initial mass and photometry bands.

        Examples
        --------
        See isoc_test.py for more details
        ```python
        parse = Parsec()
        i = Isoc(Parsec)
        isoc = i.get_isoc
        ```
        """
        return self.model.get_isoc(photsyn=photsyn, **kwargs)

    def bulk_load(self, photsyn: str, n_jobs: int = 20, **kwargs: dict) -> object:
        """
        Bulk Laod isochrones.

        Parameters
        ----------
        n_jobs : int, optional
            Default 20.
        photsyn
        kwargs : dict
            - logage_grid (tuple): start, end, step
            - mh_grid (tuple): start, end, step
        """
        self.model.bulk_load(photsyn=photsyn, n_jobs=n_jobs, **kwargs)
