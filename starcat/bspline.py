import os
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate


class Edr3LogMagUncertainty(object):
    """
    Estimate the log(mag) vs mag uncertainty for G, G_BP, G_RP based on Gaia EDR3 photometry.
    Originate from: [gaia-dr3-photometric-uncertainties](https://github.com/gaia-dpci/gaia-dr3-photometric-uncertainties)
    """

    def __init__(self, spline_param):
        """
        """
        _df = pd.read_csv(spline_param)
        splines = dict()
        splines['g'] = self.__init_spline(_df, 'knots_G', 'coeff_G')
        splines['bp'] = self.__init_spline(_df, 'knots_BP', 'coeff_BP')
        splines['rp'] = self.__init_spline(_df, 'knots_RP', 'coeff_RP')
        self.__splines = splines
        self.__nobs = {'g': 200, 'bp': 20, 'rp': 20}

    def estimate(self, band, nobs: np.array([], int) = 0, mag_range=None, mag_samples=1000):
        """
        Estimate the log(mag) vs mag uncertainty

        Parameters
        ----------
        band : str
            name of the band for which the uncertainties should be estimated (case-insentive)
        nobs : ndarray, int
            number of observations for which the uncertainties should be estimated.
            Must be a scalar integer value or an array of integer values.
        mag_range : array_like
            Magnitude range over which the spline should be evaluated.
            The default and maximum valid range is (4, 21)
        mag_samples : int
            Number evenly spaced magnitudes (over the mag_range interval) at which the splines
            will be estimated. Default: 1000

        Returns
        -------
        df : DataFrame
            Pandas dataframe with the interpolated log(mag) uncertainty vs mag.
            The magnitude column is named mag_g, mag_bp, or mag_rp depending on the requested band.
            A column for each value of nobs is provided, in the default case the column is logU_200.
        """
        band = band.lower()
        if band not in ['g', 'bp', 'rp']:
            raise ValueError(f'Unknown band: {band}')
        if mag_range is None:
            mag_range = (4., 21.)
        else:
            if mag_range[0] < 4.:
                raise ValueError(f'Uncertainties can be estimated on in the range {band}[4, 21]')
            elif mag_range[1] > 21.:
                raise ValueError(f'Uncertainties can be estimated on in the range {band}[4, 21]')
            elif mag_range[0] > mag_range[1]:
                raise ValueError('Malformed magnitude range')
        xx = np.linspace(mag_range[0], mag_range[1], mag_samples)
        __cols = self.__compute_nobs(band, xx, nobs)
        __dc = {f'mag_{band}': xx, **__cols}
        return pd.DataFrame(data=__dc)

    def __init_spline(self, df, col_knots, col_coeff):
        __ddff = df[[col_knots, col_coeff]].dropna()
        return interpolate.BSpline(__ddff[col_knots], __ddff[col_coeff], 3, extrapolate=False)

    def __compute_nobs(self, band, xx, nobs):
        if isinstance(nobs, int):
            nobs = [nobs]
        __out = dict()
        for num in nobs:
            if num < 0:
                raise ValueError(f'Number of observations should be strictly positive')
            if num == 0:
                __out[f'logU_{self.__nobs[band]:d}'] = self.__splines[band](xx)
            else:
                __out[f'logU_{num:d}'] = self.__splines[band](xx) - np.log10(np.sqrt(num) / np.sqrt(self.__nobs[band]))
        return __out
