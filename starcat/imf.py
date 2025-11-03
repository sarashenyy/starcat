import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import integrate


# Adapted from H.Monteiro, https://github.com/hektor-monteiro/OCFit/blob/master/gaiaDR2/oc_tools_padova.py

class IMF(object):
    def __init__(self, type='salpeter55'):
        """

        Parameters
        ----------
        type : str
            optional, 'chabrier03' , 'kroupa01' , 'salpeter55'
        """
        self.type = type
        # if self.type == 'kroupa01':
        #     # np.where(condition, x, y) if condition==True:x    else:y
        #     self.imf = lambda x: np.where(x < 0.5, 2 * x ** -1.3, x ** -2.3)
        # elif self.type == 'salpeter55':
        #     self.imf = lambda x: x ** -2.35
        # elif self.type == 'chabrier':
        #     self.imf = lambda x: np.where(x < 1.0, x ** -1.55, x ** -2.7)

    # def pdf_imf(self, m_i, mass_min, mass_max):
    #     mask = (m_i >= mass_min) & (m_i <= mass_max)
    #     return np.where(mask, self.imf(m_i) / integrate.quad(self.imf, mass_min, mass_max)[0], 0)

    @staticmethod
    def plot_imf(mass_min, mass_max, n_stars):
        mass = pd.DataFrame(columns=['chabrier03', 'kroupa01', 'salpeter55'], index=range(n_stars))
        mass_int = np.flip(np.logspace(np.log10(mass_min), np.log10(mass_max), 1000), axis=0)
        random_values = np.random.rand(n_stars)
        for i in range(3):
            if i == 0:
                imf_val = IMF.chabrier03(mass_int)
            elif i == 1:
                imf_val = IMF.kroupa01(mass_int)
            elif i == 2:
                imf_val = IMF.salpeter55(mass_int)
            # pdf
            pdf_imf = imf_val / integrate.trapezoid(imf_val, mass_int)
            # cdf
            cdf_imf = integrate.cumulative_trapezoid(pdf_imf, mass_int, initial=0)

            # random seed
            # r = np.random.RandomState(42)

            # mass_interp = interp1d(cum_imf, mass_int)
            # mass = mass_interp(r.rand(n_stars))
            # mass.iloc[:, i] = (interpolate.interp1d(cdf_imf, mass_int))(r.rand(n_stars))

            # using scipy.interpolate.interp1d
            # mass[mass.columns[i]] = (interpolate.interp1d(cdf_imf, mass_int))(random_values)
            # using np.interp
            mass[mass.columns[i]] = np.interp(random_values, cdf_imf, mass_int)

        # x = np.linspace(mass_min, mass_max, 10000)
        # y = self.imf(x)a
        # plt.loglog(x, y, label=self.type)  # logarithmic
        bins = 80
        hist_chabrier, _ = np.histogram(np.log10(mass.loc[:, 'chabrier03']), bins=bins)
        hist_kroupa, _ = np.histogram(np.log10(mass.loc[:, 'kroupa01']), bins=bins)
        hist_salpeter, _ = np.histogram(np.log10(mass.loc[:, 'salpeter55']), bins=bins)

        # Define bin centers
        bin_centers = 0.5 * (_[1:] + _[:-1])

        # Plot histograms as lines with different line styles
        plt.plot(bin_centers, hist_salpeter, label='Salpeter 1955', linestyle='-', linewidth=2, color='#2ca02c')
        plt.plot(bin_centers, hist_kroupa, label='Kroupa 2001', linestyle='--', linewidth=2, color='#ff7f0e')
        plt.plot(bin_centers, hist_chabrier, label='Chabrier 2003', linestyle=':', linewidth=2, color='#1f77b4')
        plt.ylim(top=0.8e6)

        # Set y-axis ticks in scientific notation
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        plt.legend()
        plt.xlabel(r'$\log{m}\ (M_\odot)$')
        plt.ylabel(r'$N$')
        plt.title('Initial Mass Function (IMF)')
        plt.show()
        return mass

    @staticmethod
    def chabrier03(mass_int, Mc=None, sigma=None, alpha=None, Mnorm=None):
        """
        NOTE1: This function is under LINEAR mass coordinate,
              and WITHOUT normalization (which is done numerically in sample()).
        NOTE2, relation between LOG mass and LINEAR mass:
              Xi(m) = (1 / (m * ln(10))) * Xi(log m)
        For mass_int <= 1: log-normal form
        Chabrier (2003) - http://adsabs.harvard.edu/abs/2003PASP..115..763C
        A = 0.158, Mc = 0.079, sigma = 0.69
        For mass_int >1: power-law form
        A = 4.43e-2, alpha = 2.3

        Parameters
        ----------
        mass_int : array_like
            Mass grid (1D numpy array)
        Mc : float
            Characteristic mass of log-normal (default 0.079 Msun)
        sigma : float
            Width of log-normal (default 0.69)
        alpha : float
            High-mass power-law slope (default 2.3)
        Mnorm : float
            Mass at which to switch from log-normal to power-law (default 1.0 Msun)

        Returns
        -------
        imf_val : np.ndarray
            IMF values on linear mass coordinate (dN/dm)
        """
        if Mc is None:
            Mc = 0.079
        if sigma is None:
            sigma = 0.69
        if alpha is None:
            alpha = 2.3
        if Mnorm is None:
            Mnorm = 1.0
        # assume mass_int is a 1D numpy array of positive floats (ascending or not doesn't matter here)
        imf_val = np.empty_like(mass_int, dtype=float)
        # use boolean mask (clearer than np.where tuple)
        mask_low = (mass_int <= Mnorm)  # low: log-normal
        mask_high = ~mask_low  # complementary: power-law

        # scaling factor to ensure continuity in shape (which will be overridden after numerical normalization).
        factor = (Mnorm ** (alpha - 1) / np.log(10)) * np.exp(
            -0.5 * ((np.log10(Mnorm) - np.log10(Mc)) / sigma) ** 2
        )

        # M <= Mnorm: log-normal, LINEAR mass coordinate
        imf_val[mask_low] = (1.0 / (mass_int[mask_low] * np.log(10))) * np.exp(
            -0.5 * ((np.log10(mass_int[mask_low]) - np.log10(Mc)) / sigma) ** 2
        )

        # M > Mnorm: scaled power-law, LINEAR mass coordinate
        imf_val[mask_high] = factor * mass_int[mask_high] ** (-alpha)

        return imf_val

    # @staticmethod
    # def kroupa01(mass_int, alpha1=None, alpha2=None):
    #     """
    #     Kroupa (2001) - https://doi.org/10.1046/j.1365-8711.2001.04022.x
    #
    #     Parameters
    #     ----------
    #     mass_int
    #     alpha1 : float
    #         Default is 1.3
    #     alpha2 : float
    #         Default is 2.3
    #
    #     Returns
    #     -------
    #
    #     """
    #     if alpha1 is None:
    #         alpha1 = 1.3
    #     if alpha2 is None:
    #         alpha2 = 2.3
    #     m_break = 0.5
    #     factor = m_break ** (alpha2 - alpha1)  # factor for scale
    #
    #     id_low = np.where(mass_int <= m_break)
    #     # upper
    #     imf_val = factor * (mass_int ** -alpha2)
    #     # lower
    #     imf_val[id_low] = mass_int[id_low] ** -alpha1
    #     return imf_val

    @staticmethod
    def kroupa01(mass_int, alpha1=None, alpha2=None, m_break=None):
        """
        Kroupa (2001) - https://doi.org/10.1046/j.1365-8711.2001.04022.x

        Parameters
        ----------
        mass_int
        alpha1 : float
            Default is 1.3
        alpha2 : float
            Default is 2.3
        m_break : float
            Default is 0.5

        Returns
        -------

        """
        if alpha1 is None:
            alpha1 = 1.3
        if alpha2 is None:
            alpha2 = 2.3
        if m_break is None:
            m_break = 0.5
        imf_val = np.empty_like(mass_int, dtype=float)
        factor = m_break ** (alpha2 - alpha1)  # factor for scale
        id_low = np.where(mass_int <= m_break)
        # upper
        imf_val = factor * (mass_int ** -alpha2)
        # lower
        imf_val[id_low] = mass_int[id_low] ** -alpha1
        return imf_val

    @staticmethod
    def salpeter55(mass_int, alpha=None):
        """

        Parameters
        ----------
        mass_int
        alpha : float
            Default is 2.35

        Returns
        -------

        """
        if alpha is None:
            alpha = 2.35
        imf_val = mass_int ** -alpha
        return imf_val

    def sample(self, n_stars, mass_min, mass_max, alpha=None, mbreak=None, seed=None, **kwargs):
        """
        Generate n stars which mass distribution obey with imf.

        Parameters
        ----------
        alpha : float / [float, float]
            alpha / [alpha1, alpha2]
        n_stars : int
            n stars
        mass_min : float
        mass_max : float

        kwargs : dict
            Optional keyword arguments:
            - Mc, sigma, Mnorm : for Chabrier03
            - mbreak : for Kroupa01
            - seed : random seed

        Returns
        -------
            list: A mass list of length n.
        """
        # np.random.seed(42)
        # mass = []
        # c = self.pdf_imf(mass_min, mass_min, mass_max)
        # while len(mass) < n_stars:
        #     m_x = np.random.uniform(low=mass_min, high=mass_max, size=n_stars)
        #     m_y = np.random.uniform(0, 1, size=n_stars)
        #     mask = m_y < self.pdf_imf(m_x, mass_min, mass_max) / c
        #     mass.extend(m_x[mask])
        # return mass[:n_stars]

        # spaced evenly on a log scale, but still LINEAR mass coordinate
        mass_int = np.flip(np.logspace(np.log10(mass_min), np.log10(mass_max), 2000), axis=0)
        if self.type == 'chabrier03':
            # extract optional parameters safely
            Mc = kwargs.get('Mc')
            sigma = kwargs.get('sigma')
            Mnorm = kwargs.get('Mnorm')
            imf_val = self.chabrier03(mass_int, Mc=Mc, sigma=sigma, alpha=alpha, Mnorm=Mnorm)
        elif self.type == 'kroupa01':
            alpha1, alpha2 = alpha
            mbreak = mbreak
            imf_val = IMF.kroupa01(mass_int, alpha1, alpha2, mbreak)
        elif self.type == 'salpeter55':
            imf_val = IMF.salpeter55(mass_int, alpha)

        # pdf, integrate.trapezoid(imf_val, mass_int)
        pdf_imf = imf_val / integrate.trapezoid(imf_val, mass_int)

        # cdf
        cdf_imf = integrate.cumulative_trapezoid(pdf_imf, mass_int, initial=0)

        # mass_interp = interp1d(cum_imf, mass_int)
        # mass = mass_interp(r.rand(n_stars))
        # r.rand()  random samples from a uniform distribution over [0, 1)

        if seed is not None:
            # random seed
            r = np.random.RandomState(seed)
            random_values = r.rand(n_stars)
        else:
            random_values = np.random.rand(n_stars)
        # using scipy.interpolate.interp1d()
        # mass = (interpolate.interp1d(cdf_imf, mass_int))(random_values)
        # using np.interp(x,xp,fp,left,right)
        # xp must be increasing : np.all(np.diff(xp)>0)
        # np.all(np.diff(cdf_imf)>0)
        mass = np.interp(random_values, cdf_imf, mass_int)
        return mass
