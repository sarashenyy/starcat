import os
from abc import ABC, abstractmethod
import joblib
import numpy as np
from berliner import CMD

from .widgets import round_to_step
from . import config


class Model(ABC):
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


class Parsec(Model):
    """
    subclass for abstract base class Model()
    """

    def __init__(self):
        self.model = "parsec"

    def get_isoc(self, photsyn, **kwargs):
        """
        Get isochrone from parsec model.
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
        logage = round_to_step(kwargs.get("logage"), step=kwargs.get("logage_step"))
        mh = round_to_step(kwargs.get("mh"), step=kwargs.get("mh_step"))
        # TODO: dm 用于确定最小质量 mass_min, 是否需要更改确定最小质量(最大光度)的方式？
        # dm = kwargs.get("dm")
        # mag_max = source["mag_max"]
        source = config.config[self.model][photsyn]
        bands = source["bands"]
        mini = source["mini"]
        label = source["label"]
        phase = source["phase"]
        isoc_dir = source["isoc_dir"]
        isoc_path = config.data_dir + isoc_dir + f"age{logage:+.2f}_mh{mh:+.2f}.joblib"

        if os.path.exists(isoc_path):
            isochrone = joblib.load(isoc_path)
        else:
            c = CMD()
            isochrone = c.get_one_isochrone(
                logage=logage, z=None, mh=mh, photsys_file=photsyn
            )
            # truncate isochrone, PMS~EAGB
            # ATTENTION! parsec use "label" to represent evolutionary phase, different from MIST("phase")
            isochrone = isochrone[
                (isochrone["label"] >= min(label)) & (isochrone["label"] <= max(label))].to_pandas()
            # add evolutionary phase info
            for i, element in enumerate(label):
                index = np.where(isochrone["label"] == element)[0]
                isochrone.loc[index, "phase"] = phase[i]
        # TODO: 将以下两行定义质量范围的代码和上述一行定义dm的代码移出Isoc类之外
        # mass_min = min(isochrone[(isochrone[bands[0]] + dm) <= mag_max][mini])
        # mass_max = max(isochrone[mini])

        # save isochrone file
        if not os.path.exists(isoc_path):
            joblib.dump(isochrone, isoc_path)
        useful_columns = ["phase", mini] + bands
        isoc = isochrone[useful_columns]
        return isoc


class MIST(Model):
    """
    subclass for abstract base class Model()
    """

    def __init__(self):
        self.model = "mist"

    def get_isoc(self, photsyn, **kwargs):
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
