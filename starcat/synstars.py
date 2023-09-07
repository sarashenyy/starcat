from abc import ABC, abstractmethod

from . import config
from .magerr import MagError


class Photsys(ABC):
    @abstractmethod
    def add_syn_photerr(self, sample_syn, **kwargs):
        pass


class GaiaEDR3(Photsys):
    def __init__(self, model):
        """

        Parameters
        ----------
        model : str
            'parsec' or 'MIST'.
        """
        self.photsys = 'gaiaEDR3'
        self.model = model
        source = config.config[self.model][self.photsys]
        self.bands = source['bands']

    def add_syn_photerr(self, sample_syn, **kwargs):
        """
        Add synthetic photoerror to synthetic star sample.

        Parameters
        ----------
        sample_syn : pd.DataFrame, cotaining [bands] cols
            Synthetic sample without photoerror.
        **kwargs :
            - med_nobs ( list[int] ): Median observation number of each band. [G, BP, RP]
        Returns
        -------
        pd.DataFrame : [bands] x [_syn]
        """
        med_nobs = kwargs.get('med_nobs')
        e = MagError(med_nobs=med_nobs, bands=self.bands)
        # g_med_err, bp_med_err, rp_med_err = e.estimate_med_photoerr(sample_syn=sample_syn)
        # sample_syn[self.bands[0] + '_err_syn'], sample_syn[self.bands[1] + '_err_syn'], sample_syn[
        #     self.bands[2] + '_err_syn'] = (
        #     g_med_err, bp_med_err, rp_med_err)
        g_syn, bp_syn, rp_syn = e(sample_syn=sample_syn)
        sample_syn[self.bands[0] + '_syn'], sample_syn[self.bands[1] + '_syn'], sample_syn[
            self.bands[2] + '_syn'] = (
            g_syn, bp_syn, rp_syn)
        return sample_syn

# class SynStars(object):
#     def __init__(self, imf, n_stars):
#         self.imf = imf
#         self.n_stars = n_stars
#
#     def __call__(self, *args, **kwargs):
#         pass
#
#     def sample_stars(self, isoc, fb, n_stars, mass_min, mass_max):
#         n_binary = int(n_stars * fb)
#         sample_syn = pd.DataFrame(np.zeros((n_stars, 1)), columns=['mass_pri'])
#         sample_syn['mass_pri'] = self.imf.sample(n_stars=n_stars, mass_min=mass_min, mass_max=mass_max)
#         # add binaries
#         # if mass_sec != NaN, then binaries
#         secindex = random.sample(list(sample_syn.index), k=n_binary)
#
#         masssec_min = min(isoc[isoc['phase'] == 'MS'][self.Mini])
#         masssec_max = max(isoc[isoc['phase'] == 'MS'][self.Mini])
