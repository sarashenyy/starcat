import random
from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import interpolate

from . import config


class BinMethod(ABC):
    @abstractmethod
    def add_binary(self, fb, n_stars, sample, isoc, imf, model, photsyn, *args):
        pass


class BinMS(BinMethod):
    """
    Secondary stars are all Main Sequence stars.
    """

    def __init__(self):
        self.method = 'MainSequenceBinary'

    def add_binary(self, fb, n_stars, sample, isoc, imf, model, photsyn, *args):
        """
        Add binaries to sample. [mass_pri] ==> [ mass x [_pri, _sec], bands x [_pri, _sec, _syn]
        Secondary stars are all Main Sequence stars.

        Parameters
        ----------
        isoc : pd.DataFrame
        imf : starcat.IMF
        fb : float
        n_stars : int
        sample : pd.DataFrame
        model : str
            'parsec' or 'MIST'
        photsyn : str
            'gaiaDR2' or 'gaiaEDR3'
        """
        masssec_min, masssec_max = define_secmass_ms(isoc=isoc, model=model, photsyn=photsyn)
        sample = add_binary_wrapper(
            fb=fb, n_stars=n_stars, sample=sample, isoc=isoc, imf=imf,
            model=model, photsyn=photsyn, masssec_min=masssec_min, masssec_max=masssec_max
        )
        return sample


class BinSimple(BinMethod):
    """
    Secondary stars have the same mass range with sample.
    """

    def __init__(self):
        self.method = 'SimpleBinary'

    def add_binary(self, fb, n_stars, sample, isoc, imf, model, photsyn, *args):
        """
        Add binaries to sample. [mass_pri] ==> [ mass x [_pri, _sec], bands x [_pri, _sec, _syn]
        Secondary stars have the same mass range with sample.

        Parameters
        ----------
        isoc : pd.DataFrame
        imf : starcat.IMF
        fb : float
        n_stars : int
        sample : pd.DataFrame
        model : str
            'parsec' or 'MIST'
        photsyn : str
            'gaiaDR2' or 'gaiaEDR3'
        *args :
            - dm (float)
        """
        dm = args
        masssec_min, masssec_max = define_secmass_simple(isoc=isoc, dm=dm, model=model, photsyn=photsyn)
        sample = add_binary_wrapper(
            fb=fb, n_stars=n_stars, sample=sample, isoc=isoc, imf=imf,
            model=model, photsyn=photsyn, masssec_min=masssec_min, masssec_max=masssec_max
        )
        return sample


def add_binary_wrapper(fb, n_stars, sample, isoc, imf, model, photsyn, masssec_min, masssec_max):
    """
    Add binaries to sample. [mass_pri] ==> [ mass x [_pri, _sec], bands x [_pri, _sec, _syn]

    Parameters
    ----------
    masssec_max : float
    masssec_min : float
    isoc : pd.DataFrame
    imf : starcat.IMF
    fb : float
    n_stars : int
    sample : pd.DataFrame
    model : str
        'parsec' or 'MIST'
    photsyn : str
        'gaiaDR2' or 'gaiaEDR3'
    """
    source = config.config[model][photsyn]
    mini = source['Mini']
    bands = source['bands']
    phase = source['phase']

    # add binaries
    # if mass_sec != NaN, then binaries
    n_binary = int(n_stars * fb)
    secindex = random.sample(list(sample.index), k=n_binary)
    sample.loc[secindex, 'mass_sec'] = imf.sample(n_stars=n_binary, mass_min=masssec_min, mass_max=masssec_max)

    # add mag for each band
    for band in bands:
        # piecewise mass_mag relation
        id_cut = phase.index('CHEB')
        range1 = phase[0:id_cut]
        range2 = phase[id_cut:]
        mass_mag_1 = interpolate.interp1d(
            isoc[isoc['phase'].isin(range1)][mini],
            isoc[isoc['phase'].isin(range1)][band], fill_value='extrapolate')
        # if isoc has phase up to CHEB
        if isoc['phase'].isin(range2).any():
            mass_mag_2 = interpolate.interp1d(
                isoc[isoc['phase'].isin(range2)][mini],
                isoc[isoc['phase'].isin(range2)][band], fill_value='extrapolate')
            mass_cut = min(isoc[isoc['phase'].isin(range2)][mini])
        # else
        else:
            mass_mag_2 = mass_mag_1
            mass_cut = max(isoc[isoc['phase'].isin(range1)][mini])
        # add mag for primary(including single) & secondary star
        for m in ['pri', 'sec']:
            sample.loc[sample['mass_%s' % m] < mass_cut, '%s_%s' % (band, m)] = (
                mass_mag_1(sample[sample['mass_%s' % m] < mass_cut]['mass_%s' % m]))
            sample.loc[sample['mass_%s' % m] >= mass_cut, '%s_%s' % (band, m)] = (
                mass_mag_2(sample[sample['mass_%s' % m] >= mass_cut]['mass_%s' % m]))
        # add syn mag (for binaries, syn = f(pri,sec) )
        sample['%s_syn' % band] = sample['%s_pri' % band]
        sample.loc[secindex, '%s_syn' % band] = (
                -2.5 * np.log10(pow(10, -0.4 * sample.loc[secindex, '%s_pri' % band])
                                + pow(10, -0.4 * sample.loc[secindex, '%s_sec' % band])
                                )
        )
    return sample


def define_secmass_ms(isoc, model, photsyn):
    source = config.config[model][photsyn]
    mini = source['Mini']
    # Key Point! secondary stars are all Main Sequence stars
    masssec_min = min(isoc[isoc['phase'] == 'MS'][mini])
    masssec_max = max(isoc[isoc['phase'] == 'MS'][mini])
    return masssec_min, masssec_max


def define_secmass_simple(isoc, dm, model, photsyn):
    source = config.config[model][photsyn]
    mini = source['Mini']
    mag = source['mag']
    mag_max = source['mag_max']
    masssec_min = min(
        isoc[(isoc[mag] + dm) <= mag_max][mini]
    )
    masssec_max = max(isoc[mini])
    return masssec_min, masssec_max
