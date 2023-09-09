import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate


class IMF(object):
    def __init__(self, type='kroupa01'):
        self.type = type
        if self.type == 'kroupa01':
            # np.where(condition, x, y) if condition==True:x    else:y
            self.imf = lambda x: np.where(x < 0.5, 2 * x ** -1.3, x ** -2.3)  # x2 for scale
        elif self.type == 'salpeter55':
            self.imf = lambda x: x ** -2.35
        elif self.type == 'chabrier':
            self.imf = lambda x: np.where(x < 1.0, x ** -1.55, x ** -2.7)

    def pdf_imf(self, m_i, mass_min, mass_max):
        mask = (m_i >= mass_min) & (m_i <= mass_max)
        return np.where(mask, self.imf(m_i) / integrate.quad(self.imf, mass_min, mass_max)[0], 0)

    def plot_imf(self, mass_min, mass_max):
        x = np.linspace(mass_min, mass_max, 1000)
        y = self.imf(x)
        plt.loglog(x, y, label=self.type)  # logarithmic
        plt.legend()
        plt.xlabel('Mass [Solar mass]')
        plt.ylabel('IMF')
        plt.title('Initial Mass Function (IMF)')
        plt.show()

    def sample(self, n_stars, mass_min, mass_max):
        """
        Generate n stars which mass distribution obey with imf.

        Parameters
        ----------
        n_stars : int
            n stars
        mass_min : float
        mass_max : float

        Returns
        -------
            list: A mass list of length n.
        """
        mass = []
        c = self.pdf_imf(mass_min, mass_min, mass_max)
        while len(mass) < n_stars:
            m_x = np.random.uniform(low=mass_min, high=mass_max, size=n_stars)
            m_y = np.random.uniform(0, 1, size=n_stars)
            mask = m_y < self.pdf_imf(m_x, mass_min, mass_max) / c
            mass.extend(m_x[mask])
        return mass[:n_stars]
