import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import integrate


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
    def chabrier03(mass_int, alpha=None):
        """
        For mass_int <= 1.
        Chabrier (2003) - http://adsabs.harvard.edu/abs/2003PASP..115..763C
        For mass_int >1.
        A = 4.43e-2, alpha = 2.3

        Parameters
        ----------
        mass_int
        alpha :

        Returns
        -------

        """
        if alpha is None:
            alpha = 2.3
        id_low = np.where(mass_int <= 1.)
        imf_val = 4.43e-2 * mass_int ** (-alpha)
        imf_val[id_low] = 0.158 * np.exp(-0.5 *
                                         (np.log10(mass_int[id_low]) - np.log10(0.08)) ** 2 / 0.69 ** 2)
        return imf_val

    @staticmethod
    def kroupa01(mass_int, alpha1=None, alpha2=None):
        """
        Kroupa (2001) - https://doi.org/10.1046/j.1365-8711.2001.04022.x

        Parameters
        ----------
        mass_int
        alpha1 : float
            Default is 1.3
        alpha2 : float
            Default is 2.3

        Returns
        -------

        """
        if alpha1 is None:
            alpha1 = 1.3
        if alpha2 is None:
            alpha2 = 2.3
        m_break = 0.5
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

    def sample(self, n_stars, mass_min, mass_max, alpha=None, seed=None):
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

        # spaced evenly on a log scale
        mass_int = np.flip(np.logspace(np.log10(mass_min), np.log10(mass_max), 1000), axis=0)
        if self.type == 'chabrier03':
            imf_val = IMF.chabrier03(mass_int, alpha)
        elif self.type == 'kroupa01':
            alpha1, alpha2 = alpha
            imf_val = IMF.kroupa01(mass_int, alpha1, alpha2)
        elif self.type == 'salpeter55':
            imf_val = IMF.salpeter55(mass_int, alpha)

        # pdf
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
