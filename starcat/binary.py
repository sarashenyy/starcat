from abc import ABC, abstractmethod

import numpy as np

from . import config


# from .logger import log_time
# import jax.numpy as jnp
# from .interp_cython import interp_cython

class BinMethod(ABC):
    @abstractmethod
    def add_binary(self, fb, n_stars, sample, isoc, imf, model, photsyn, *args, **kwargs):
        """
        Add binaries to sample. [mass_pri] ==> [ mass x [_pri, _sec], bands x [_pri, _sec], bands ]
        Secondary stars are all Main Sequence stars.

        Parameters
        ----------
        fb
        n_stars
        sample
        isoc
        imf
        model
        photsyn
        args

        Returns
        -------

        """
        pass


class BinMS(BinMethod):
    """
    Secondary stars are all Main Sequence stars.
    """

    def __init__(self):
        self.method = 'MainSequenceBinary'

    # @log_time
    def add_binary(self, fb, n_stars, sample, isoc, imf, model, photsyn, *args, **kwargs):
        """
        Add binaries to sample. [mass_pri] ==> [ mass x [_pri, _sec], bands x [_pri, _sec], bands ]
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
        sample = add_secmass_simple(
            fb=fb, n_stars=n_stars, sample=sample, imf=imf,
            masssec_min=masssec_min, masssec_max=masssec_max
        )
        sample = add_companion_mag(
            sample=sample, isoc=isoc,
            model=model, photsyn=photsyn
        )
        return sample


class BinSimple(BinMethod):
    """
    Secondary stars have the same mass range with sample.
    """

    def __init__(self):
        self.method = 'SimpleBinary'

    # @log_time
    def add_binary(self, fb, n_stars, sample, isoc, imf, model, photsyn, *args, **kwargs):
        """
        Add binaries to sample. [mass_pri] ==> [ mass x [_pri, _sec], bands x [_pri, _sec], bands ]
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
        """
        masssec_min, masssec_max = define_secmass_simple(isoc=isoc, model=model, photsyn=photsyn)
        sample = add_secmass_simple(
            fb=fb, n_stars=n_stars, sample=sample, imf=imf,
            masssec_min=masssec_min, masssec_max=masssec_max
        )
        sample = add_companion_mag(
            sample=sample, isoc=isoc,
            model=model, photsyn=photsyn
        )
        return sample


class BinMRD(BinMethod):
    """
    add condition: q > 0.09/primass
    sample secondary mass from distribution q^gamma(default gamma=0)
    """

    def __init__(self):
        self.method = 'BinMRD'

    # @log_time
    def add_binary(self, fb, n_stars, sample, isoc, imf, model, photsyn, *args, **kwargs):
        gamma = kwargs.get('gamma')
        sample = add_secmass_MRD(fb=fb, n_stars=n_stars, sample=sample, gamma=gamma)
        sample = add_companion_mag(
            sample=sample, isoc=isoc,
            model=model, photsyn=photsyn
        )
        return sample


class BinCusp(BinMethod):
    """
    mass ratio distribution is uniform + cusp
    PDF:
    f(q) = C1,  0 <= q < 0.9
    f(q) = C2 = beta * C1,  0.9 <= q <= 1
    C1 and C2 are constants, with C2 = beta * C1.
    """

    def __init__(self):
        self.method = 'BinCusp'

    def add_binary(self, fb, n_stars, sample, isoc, imf, model, photsyn, *args, **kwargs):
        beta = kwargs.get('beta')
        sample = add_secmass_cusp(fb=fb, n_stars=n_stars, sample=sample, beta=beta)
        sample = add_companion_mag(
            sample=sample, isoc=isoc,
            model=model, photsyn=photsyn
        )
        return sample


class BinCuspGamma(BinMethod):
    """
    mass ratio distribution is q^gamma + cusp(q=(0.95,1] uniform)
    parameters : fb, ftwin(q=(0.95,1]), gamma
    """

    def __init__(self):
        self.method = 'BinCuspGamma'

    def add_binary(self, fb, n_stars, sample, isoc, imf, model, photsyn, *args, **kwargs):
        gamma = kwargs.get('gamma')
        ftwin = kwargs.get('ftwin')
        sample = add_secmass_cusp_gamma(fb=fb, n_stars=n_stars, sample=sample,
                                        ftwin=ftwin, gamma=gamma)
        sample = add_companion_mag(
            sample=sample, isoc=isoc,
            model=model, photsyn=photsyn
        )
        return sample


def sample_q_gamma(qmin, gamma=0, q_threshold=1):
    """
    PDF: f(q) = q^gamma, a<q<b (a=0.09/mass_pri)
    CDF: $F(q) = \int_a^q f(x) dx = \int_a^q x^\gamma dx = \frac{1}{\gamma+1} (q^{\gamma+1} - a^{\gamma+1})$
    reverse CDF: u = F(q), q = F^(-1)(u)
    DEFAULT: uniform

    Parameters
    ----------
    qmin : np.array
        qmin = 0.09 / mass_pri
    qmax : np.array
        default is 1
    gamma : float
        default is 0
    Returns
    -------
    np.array
    """
    num = len(qmin)
    # qmax = np.ones(num)
    qmax = np.full(num, q_threshold)

    if gamma != -1:
        # When gamma is not -1, use the inverse CDF method
        r = np.random.uniform(0, 1 + 1e-4, num)
        q = ((qmax ** (gamma + 1) - qmin ** (gamma + 1)) * r + qmin ** (gamma + 1)) ** (1 / (gamma + 1))
    else:
        # When gamma is -1, use the specific form for this case
        r = np.random.uniform(0, 1 + 1e-4, num)
        q = qmin * (qmax / qmin) ** r
    return q


def sample_q_cusp(qmin, beta=2, q_threshold=0.95):
    """
    PDF:
    f(q) = C1,  0 <= q < 0.9
    f(q) = C2 = beta * C1,  0.9 <= q <= 1
    C1 and C2 are constants, with C2 = beta * C1.
    DEFAULT: uniform + cusp

    Parameters
    ----------
    beta : float
        The ratio of the densities C2/C1
    Returns
    -------
    np.array
        Array of sampled q values
    """
    num = len(qmin)
    qmax = np.ones(num)

    # Determine the area under each section of the PDF
    area_low = q_threshold * 1  # Since C1 is a constant, area_low = C1 * (0.95 - 0) = 0.95 * C1
    area_high = 0.1 * beta  # Since C2 = beta * C1, area_high = C2 * (1 - 0.95) = 0.1 * beta * C1
    total_area = area_low + area_high

    # Normalize the areas to probabilities
    prob_low = area_low / total_area
    prob_high = area_high / total_area
    # Generate uniform random numbers to decide which region to sample from
    q_split = np.random.uniform(0, 1, num)
    # Initialize arrays
    q = np.zeros(num)
    # Sampling for the low region [0, 0.9)
    low_mask = q_split < prob_low
    num_low = np.sum(low_mask)
    q[low_mask] = np.random.uniform(0, q_threshold, num_low)
    # Sampling for the high region [0.9, 1]
    high_mask = ~low_mask
    num_high = np.sum(high_mask)
    q[high_mask] = np.random.uniform(q_threshold, 1, num_high)
    return q

def add_secmass_MRD(fb, n_stars, sample, gamma=None):
    """
    Parameters
    ----------
    fb
    n_stars
    sample : pd.DataFrame
        [mass_pri]
    gamma : float
        PDF(q) = q^gamma
    Returns
    -------
    pd.DataFrames : [mass_pri, mass_sec, q]
    """
    # randomly choose binaries
    # if mass_sec != Nan, then binariesss
    n_binary = int(n_stars * fb)
    # np.random.seed(42)
    # no duplication! np.random.choice() will return duplicate index

    # secindex = random.sample(list(sample.index), n_binary)
    secindex = np.random.choice(sample.index, n_binary, replace=False)  # maybe faster -43.37ms
    mass_pri = sample.loc[secindex, 'mass_pri'].to_numpy()
    qmin = 0.09 / mass_pri
    if gamma is None:  # gamma = 0, default!
        qs = sample_q_gamma(qmin=qmin)
    else:  # gamma as free parameter
        qs = sample_q_gamma(qmin=qmin, gamma=gamma)
    sample.loc[secindex, 'mass_sec'] = qs * mass_pri
    sample.loc[secindex, 'q'] = qs
    return sample


def add_secmass_cusp(fb, n_stars, sample, beta=None):
    """

    Parameters
    ----------
    fb
    n_stars
    sample : pd.DataFrame
        [mass_pri]
    beta :   float
        f(q) = C1,  0 <= q < 0.9; f(q) = C2 = beta * C1,  0.9 <= q <= 1

    Returns
    -------
    pd.DataFrames : [mass_pri, mass_sec, q]
    """
    n_binary = int(n_stars * fb)

    # secindex = random.sample(list(sample.index), n_binary)
    secindex = np.random.choice(sample.index, n_binary, replace=False)  # maybe faster -43.37ms
    mass_pri = sample.loc[secindex, 'mass_pri'].to_numpy()
    qmin = 0.09 / mass_pri
    if beta is None:  # beta = 2, default!
        qs = sample_q_cusp(qmin=qmin)
    else:  # beta as free parameter
        qs = sample_q_cusp(qmin=qmin, beta=beta)
    sample.loc[secindex, 'mass_sec'] = qs * mass_pri
    sample.loc[secindex, 'q'] = qs
    return sample


def add_secmass_cusp_gamma(fb, n_stars, sample, ftwin, gamma):
    """
    ftwin: fraction of twin binaries, q = (q_thres, 1); uniform q
    fb: fraction of binary stars, q < q_thres; q^gamma

    Parameters
    ----------
    fb
    n_stars
    sample
    ftwin
    gamma

    Returns
    -------

    """
    q_thres = 0.95
    n_binary = int(n_stars * fb)  # q = (qmin, q_threshold)
    n_twin = int(n_stars * ftwin)  # q = (q_threshold, 1)
    secindex = np.random.choice(sample.index, n_binary + n_twin, replace=False)
    secindex_b = secindex[:n_binary]  # n_binary
    secindex_t = secindex[n_binary:]  # n_twin

    # binary, q = (0, 0.95) ; q^gamma distribution
    mass_pri_b = sample.loc[secindex_b, 'mass_pri'].to_numpy()
    qmin_b = 0.09 / mass_pri_b
    qs_binary = sample_q_gamma(qmin=qmin_b, gamma=gamma, q_threshold=q_thres)
    sample.loc[secindex_b, 'mass_sec'] = qs_binary * mass_pri_b
    sample.loc[secindex_b, 'q'] = qs_binary

    # twin binary, q = (0.95, 1) ; uniform distribution, gamma=0
    mass_pri_t = sample.loc[secindex_t, 'mass_pri'].to_numpy()
    qmin_t = np.full(n_twin, q_thres)
    qs_twin = sample_q_gamma(qmin=qmin_t, gamma=0, q_threshold=1)
    sample.loc[secindex_t, 'mass_sec'] = qs_twin * mass_pri_t
    sample.loc[secindex_t, 'q'] = qs_twin
    return sample

def add_secmass_simple(fb, n_stars, sample, imf, masssec_min, masssec_max, alpha=None):
    """

    Parameters
    ----------
    fb
    n_stars
    sample : pd.DataFrame
        [mass_pri]
    imf
    masssec_min
    masssec_max

    Returns
    -------
    pd.DataFrames : [mass_pri, mass_sec]
    """
    # randomly choose binaries
    # if mass_sec != Nan, then binaries
    n_binary = int(n_stars * fb)
    # np.random.seed(42)
    # # no duplication! np.random.choice() will return duplicate index

    # secindex = random.sample(list(sample.index), n_binary)
    secindex = np.random.choice(sample.index, n_binary, replace=False)
    sample.loc[secindex, 'mass_sec'] = imf.sample(n_stars=n_binary, mass_min=masssec_min,
                                                  mass_max=masssec_max, alpha=alpha)
    return sample


# @log_time
def add_companion_mag(sample, isoc, model, photsyn):
    """
    Add Mag for given Mass.
    [mass_pri, mass_sec] ==> [ mass x [_pri, _sec], bands x [_pri, _sec], bands ]

    Parameters
    ----------
    isoc : pd.DataFrame
    sample : pd.DataFrame
    model : str
        'parsec' or 'MIST'
    photsyn : str
        'gaiaDR2' or 'gaiaEDR3'
    """
    source = config.config[model][photsyn]
    mini = source['mini']
    bands = source['bands']
    # ! NO NEEDS to interpolate using piecewise, therefore NO NEEDS to use 'phase'!
    # phase = source['phase']

    # # add binaries
    # # if mass_sec != NaN, then binaries
    # n_binary = int(n_stars * fb)
    # # secindex = random.sample(list(sample.index), k=n_binary)
    # np.random.seed(42)
    # secindex = np.random.choice(list(sample.index), size=n_binary)
    # sample.loc[secindex, 'mass_sec'] = imf.sample(n_stars=n_binary, mass_min=masssec_min, mass_max=masssec_max)

    secindex = sample.dropna(subset='mass_sec').index
    # ! NO NEEDS to interpolate using piecewise!
    # add mag for each band
    # for band in bands:
    #     # piecewise mass_mag relation
    #     id_cut = phase.index('CHEB')
    #     range1 = phase[0:id_cut]
    #     range2 = phase[id_cut:]
    #
    #     mass_mag_1 = interpolate.interp1d(
    #         isoc[isoc['phase'].isin(range1)][mini],
    #         isoc[isoc['phase'].isin(range1)][band], fill_value='extrapolate')
    #     # if isoc has phase up to CHEB
    #     if isoc['phase'].isin(range2).any():
    #         mass_mag_2 = interpolate.interp1d(
    #             isoc[isoc['phase'].isin(range2)][mini],
    #             isoc[isoc['phase'].isin(range2)][band], fill_value='extrapolate')
    #         mass_cut = min(isoc[isoc['phase'].isin(range2)][mini])
    #     # else
    #     else:
    #         mass_mag_2 = mass_mag_1
    #         mass_cut = max(isoc[isoc['phase'].isin(range1)][mini])
    #
    #     # add mag for primary(including single) & secondary star
    #     for m in ['pri', 'sec']:
    #         sample.loc[sample['mass_%s' % m] < mass_cut, '%s_%s' % (band, m)] = (
    #             mass_mag_1(sample[sample['mass_%s' % m] < mass_cut]['mass_%s' % m]))
    #         sample.loc[sample['mass_%s' % m] >= mass_cut, '%s_%s' % (band, m)] = (
    #             mass_mag_2(sample[sample['mass_%s' % m] >= mass_cut]['mass_%s' % m]))
    #     # add syn mag (for binaries, syn = f(pri,sec) )
    #     sample[band] = sample['%s_pri' % band]
    #     sample.loc[secindex, band] = (
    #             -2.5 * np.log10(pow(10, -0.4 * sample.loc[secindex, '%s_pri' % band])
    #                             + pow(10, -0.4 * sample.loc[secindex, '%s_sec' % band])
    #                             )
    #     )

    # add mag for each band, without using piecewise
    mini = isoc[mini].to_numpy()
    mpri = sample['mass_pri'].to_numpy()
    msec = sample['mass_sec'].to_numpy()
    for _ in bands:
        mag = isoc[_].to_numpy()
        # ! make sure np.all(np.diff(isoc[mini])>0) is True!
        sample.loc[:, '%s_pri' % _] = np.interp(mpri, mini, mag)
        sample.loc[:, '%s_sec' % _] = np.interp(msec, mini, mag)
        # using cython
        # sample.loc[:, '%s_pri' % _] = interp_cython(mpri, mini, mag)
        # sample.loc[:, '%s_sec' % _] = interp_cython(msec, mini, mag)

        # add syn mag (for binaries, syn = f(pri,sec) )
        sample[_] = sample['%s_pri' % _]

        # sample.loc[secindex, _] = (
        #         -2.5 * np.log10(pow(10, -0.4 * sample.loc[secindex, '%s_pri' % _])
        #                         + pow(10, -0.4 * sample.loc[secindex, '%s_sec' % _])
        #                         )
        # )
        mag1 = sample.loc[secindex, '%s_pri' % _].to_numpy()  # pandas.core.series.Series -> numpy.ndarray
        mag2 = sample.loc[secindex, '%s_sec' % _].to_numpy()
        sample.loc[secindex, _] = combine_mag(mag1, mag2)

    # j_mini = jnp.array(isoc[mini].to_numpy())
    # j_mpri = jnp.array(sample['mass_pri'].to_numpy())
    # j_msec = jnp.array(sample['mass_sec'].to_numpy())
    # for _ in bands:
    #     j_mag = jnp.array(isoc[_].to_numpy())
    #     # ! make sure np.all(np.diff(isoc[mini])>0) is True!
    #     sample.loc[:, '%s_pri' % _] = jnp.interp(j_mpri, j_mini, j_mag)
    #     sample.loc[:, '%s_sec' % _] = jnp.interp(j_msec, j_mini, j_mag)
    #
    #     # add syn mag (for binaries, syn = f(pri,sec) )
    #     sample[_] = sample['%s_pri' % _]
    #
    #     # sample.loc[secindex, _] = (
    #     #         -2.5 * np.log10(pow(10, -0.4 * sample.loc[secindex, '%s_pri' % _])
    #     #                         + pow(10, -0.4 * sample.loc[secindex, '%s_sec' % _])
    #     #                         )
    #     # )
    #     mag1 = sample.loc[secindex, '%s_pri' % _].to_numpy()  # pandas.core.series.Series -> numpy.ndarray
    #     mag2 = sample.loc[secindex, '%s_sec' % _].to_numpy()
    #     sample.loc[secindex, _] = combine_mag(mag1, mag2)

    return sample


# import numba as nb
# @nb.njit
# def interp_numba():

def combine_mag(mag1, mag2):
    flux1 = 10 ** (-0.4 * mag1)
    flux2 = 10 ** (-0.4 * mag2)
    combined_flux = flux1 + flux2
    combined_mag = -2.5 * np.log10(combined_flux)
    return combined_mag


def define_secmass_ms(isoc, model, photsyn):
    # TODO: may should be discarded. otherwise should reconsider the use of 'phase'
    source = config.config[model][photsyn]
    mini = source['mini']
    # Key Point! secondary stars are all Main Sequence stars
    masssec_min = min(isoc[isoc['phase'] == 'MS'][mini])
    masssec_max = max(isoc[isoc['phase'] == 'MS'][mini])
    return masssec_min, masssec_max


def define_secmass_simple(isoc, model, photsyn):
    source = config.config[model][photsyn]
    mini = source['mini']
    # mag = source['mag']  # list
    bands = source['bands']
    band_max = [x + 0.5 for x in source['band_max']]  # list 也许以后会是个坑
    masssec_max = max(isoc[mini])
    # if len(mag) == 1:
    #     try:
    #         masssec_min = min(
    #             isoc[(isoc[mag[0]]) <= band_max[0]][mini]
    #         )
    #     except ValueError:
    #         print('Do not have any stars brighter than the mag range!!')
    #
    # elif len(mag) > 1:
    #     masssec_min = min(
    #         isoc[(isoc[mag[0]]) <= band_max[0]][mini]
    #     )
    #     for i in range(len(mag)):
    #         aux_min = min(
    #             isoc[(isoc[mag[i]]) <= band_max[i]][mini]
    #         )
    #         if aux_min < masssec_min:
    #             masssec_min = aux_min
    aux_list = []
    for i in range(len(bands)):
        # synthetic Mini range is slightly larger than the observed for the consideration of binary and photerror
        condition = isoc[bands[i]] <= band_max[i]
        filtered_isoc = isoc[condition]

        if not filtered_isoc.empty:
            aux_min = min(filtered_isoc[mini])
            aux_list.append(aux_min)
    # masssec_min = min(aux_list)
    masssec_min = 0.1

    return masssec_min, masssec_max
