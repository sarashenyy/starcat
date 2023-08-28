import logging
import os
import random
import time
from functools import wraps

import joblib
import numpy as np
import pandas as pd
from berliner import CMD
from scipy import integrate
from scipy import interpolate

from .magerr import MagError

# cofiguring log output to logfile
module_dir = os.path.dirname(__file__)
filename = os.path.basename(__file__)
filename_without_extension = os.path.splitext(filename)[0]
log_file = os.path.join(module_dir, '..', 'logs', f'{filename_without_extension}.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    filemode='w')


def log_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        logging.info(f"time init() : {run_time:.4f} s")
        return result

    return wrapper


def round_to_step(arr, step):
    return np.round(arr / step) * step


class SynStars(object):
    """
    from isochrone mock CMD

    Attributes
    ----------
    photsyn :
    n_obs :
    med_nobs :
    isochrones_dir :
    bands : list[str]
        Photometric band columns in isochrone table.
    Mini : string
        Column name of initial mass in isochrone table.
    mass_min : float
        Minimum Initial Mass for synthetic stars.
    mass_max : float
        Maximum Initial Mass for synthetic stars.
    phase : list[str]
        Stellar evolution phase.
    imf : function
        Initial Mass Function.

    """

    def __init__(self, photsyn=None, sample_obs=None, isochrones_dir=None):
        # start_time = time.time()
        if photsyn is None:
            photsyn = "gaiaDR2"
        # TODO: better give sample_obs!!。 another way is untested
        self.n_syn = None
        self.phase = None
        self.mass_max = None
        self.mass_min = None
        self.Mini = None
        self.bands = None
        self.imf = None
        self.photsyn = photsyn
        if sample_obs is not None:
            # TODO: necessary? add sample_obs=None
            self.n_obs = len(sample_obs)
            if self.photsyn == "gaiaDR2":
                self.med_nobs = MagError.extract_med_nobs(sample_obs)
        if isochrones_dir:
            self.isochrones_dir = isochrones_dir
        self.set_imf()
        # TODO: adaptive c_grid, m_grid
        # end_time = time.time()
        # run_time = end_time - start_time
        # logging.info(f"time init() : {run_time:.4f} s")

    def get_isochrone(self, model="parsec", **kwargs):
        """
        Get isochrone from Web or local.

        Parameters
        ----------
        model: str, optional
            The model to use for the isochrone (default is "parsec").
        **kwargs: dict
            Additional keyword arguments.

        Returns
        -------
        isochrone: pandas.DataFrame
            A truncated isochrone containing stellar evolutionary phases from PMS to EAGB.
            The isochrone is also saved to `isochrone_path`.
        """
        start_time = time.time()
        logage = round_to_step(kwargs['logage'], step=kwargs['logage_step'])
        mh = round_to_step(kwargs['mh'], step=kwargs['mh_step'])
        dm = kwargs['dm']
        if self.isochrones_dir:
            isochrone_path = self.isochrones_dir + f"age{logage:+.2f}_mh{mh:+.2f}.joblib"
        # get isochrone file
        if os.path.exists(isochrone_path):
            isochrone = joblib.load(isochrone_path)
        else:
            if model == 'parsec' and self.photsyn == 'gaiaDR2':
                c = CMD()  # initialize berliner CMD
                isochrone = c.get_one_isochrone(
                    logage=logage, z=None, mh=mh,
                    photsys_file=self.photsyn)
                # truncate isochrone, PMS ~ EAGB
                isochrone = isochrone[(isochrone['label'] >= 0) & (isochrone['label'] <= 7)].to_pandas()
            elif model == 'mist':
                print("wait for developing")
                pass
        # extract info from isochrone
        if self.photsyn == 'gaiaDR2':
            mag_max = 18
            self.bands = ['Gmag', 'G_BPmag', 'G_RPmag']
            self.Mini = 'Mini'
            self.mass_min = min(isochrone[(isochrone['Gmag'] + dm) <= mag_max][self.Mini])
            self.mass_max = max(isochrone[self.Mini])
            # add evolutionary phase info
            self.phase = ['PMS', 'MS', 'SGB', 'RGB', 'CHEB', 'CHEB_b', 'CHEB_r', 'EAGB']
            for i in range(8):
                index = np.where(isochrone['label'] == i)[0]
                isochrone.loc[index, 'phase'] = self.phase[i]
        # save isochrone file
        if not os.path.exists(isochrone_path):
            joblib.dump(isochrone, isochrone_path)
        # a truncated isochrone (label), so mass_min and mass_max defined
        end_time = time.time()
        run_time = end_time - start_time
        logging.info(f"time get_isochrone() : {run_time:.4f} s")
        return isochrone

    def set_imf(self, imf='kroupa_2001', alpha=2):
        if imf == 'kroupa_2001':
            # np.where(condition, x, y) if condition==True:x    else:y
            self.imf = lambda x: np.where(x < 0.5, 2 * x ** -1.3, x ** -2.3)  # x2 for scale
        elif imf == 'salpeter':
            self.imf = lambda x: x ** -2.35
        elif imf == 'chabrier':
            self.imf = lambda x: np.where(x < 1.0, x ** -1.55, x ** -2.7)

    def pdf_imf(self, m_i, mass_min, mass_max):
        mask = (m_i >= mass_min) & (m_i <= mass_max)
        return np.where(mask, self.imf(m_i) / integrate.quad(self.imf, mass_min, mass_max)[0], 0)

    def random_imf(self, n, mass_min, mass_max):
        """Generate n stars which mass distribution obey with imf.

        Returns:
            A sample of n mass.
        """
        start_time = time.time()
        mass = []
        c = self.pdf_imf(mass_min, mass_min, mass_max)
        while len(mass) < n:
            m_x = np.random.uniform(low=mass_min, high=mass_max, size=n)
            m_y = np.random.uniform(0, 1, size=n)
            mask = m_y < self.pdf_imf(m_x, mass_min, mass_max) / c
            mass.extend(m_x[mask])
        end_time = time.time()
        run_time = end_time - start_time
        logging.info(f"time random_imf() : {run_time:.4f} s")
        return mass[:n]

    def sample_imf(self, fb, isochrone, n_stars, method='simple'):
        """

        Returns
        -------
        sample_syn :
        """
        start_time = time.time()
        n_binary = int(n_stars * fb)
        sample_syn = pd.DataFrame(np.zeros((n_stars, 1)), columns=['mass_pri'])
        sample_syn['mass_pri'] = self.random_imf(n_stars, self.mass_min, self.mass_max)
        # add binaries
        # if mass_sec != NaN, then binaries
        secindex = random.sample(list(sample_syn.index), k=n_binary)
        # hypothesis : secondary stars are MS only !
        if method == 'hypothesis':
            masssec_min = min(isochrone[isochrone['phase'] == 'MS'][self.Mini])
            masssec_max = max(isochrone[isochrone['phase'] == 'MS'][self.Mini])
        # without hypothesis : secondary mass range the same with primary star (ex. RGB + RGB exists)
        elif method == 'simple':
            masssec_min = self.mass_min
            masssec_max = self.mass_max
        sample_syn.loc[secindex, 'mass_sec'] = self.random_imf(n_binary, masssec_min, masssec_max)
        # add mag for each band
        for band in self.bands:
            # piecewise mass_mag relation
            id_cut = self.phase.index('CHEB')
            range1 = self.phase[0:id_cut]
            range2 = self.phase[id_cut:]
            mass_mag_1 = interpolate.interp1d(
                isochrone[isochrone['phase'].isin(range1)][self.Mini],
                isochrone[isochrone['phase'].isin(range1)][band], fill_value='extrapolate')
            # if isochrone has phase up to CHEB
            if isochrone['phase'].isin(range2).any():
                mass_mag_2 = interpolate.interp1d(
                    isochrone[isochrone['phase'].isin(range2)][self.Mini],
                    isochrone[isochrone['phase'].isin(range2)][band], fill_value='extrapolate')
                mass_cut = min(isochrone[isochrone['phase'].isin(range2)][self.Mini])
            # else
            else:
                mass_mag_2 = mass_mag_1
                mass_cut = max(isochrone[isochrone['phase'].isin(range1)][self.Mini])
            # add mag for primary(including single) & secondary star
            for m in ['pri', 'sec']:
                sample_syn.loc[sample_syn['mass_%s' % m] < mass_cut, '%s_%s' % (band, m)] = (
                    mass_mag_1(sample_syn[sample_syn['mass_%s' % m] < mass_cut]['mass_%s' % m]))
                sample_syn.loc[sample_syn['mass_%s' % m] >= mass_cut, '%s_%s' % (band, m)] = (
                    mass_mag_2(sample_syn[sample_syn['mass_%s' % m] >= mass_cut]['mass_%s' % m]))
                # add syn mag (for binaries, syn = f(pri,sec) )
            sample_syn['%s_syn' % band] = sample_syn['%s_pri' % band]
            sample_syn.loc[secindex, '%s_syn' % band] = (
                    -2.5 * np.log10(pow(10, -0.4 * sample_syn.loc[secindex, '%s_pri' % band])
                                    + pow(10, -0.4 * sample_syn.loc[secindex, '%s_sec' % band])))
        end_time = time.time()
        run_time = end_time - start_time
        logging.info(f"time sample_imf() : {run_time:.4f} s")
        return sample_syn
        # a sample of mock single & binaries stars [ mass x [_pri, _sec], self.bands x [_pri, _sec, _syn]

    def estimate_syn_photerror(self, sample_syn, **kwargs):
        """

        Parameters
        ----------
        sample_syn
        kwargs

        Returns
        -------
        sample_syn: pandas.DataFrame
            A sample of mock stars WITH band error added.
            [ mass x [_pri, _sec], bands x [_pri, _sec, _syn, _err_syn]

        """
        # bands=['Gmag','G_BPmag','G_RPmag'],bands_err=['Gmag_err','G_BPmag_err','G_RPmag_err']
        start_time = time.time()
        # add photoerror for sample_syn
        # method 1 , wait for developing : interpolate & scale Edr3LogMagUncertainty
        if self.photsyn == "gaiaDR2":
            if self.med_nobs is not None:
                # e = MagError(med_nobs=self.med_nobs,bands=[_+'_syn',for _ in self.bands])
                e = MagError(med_nobs=self.med_nobs, bands=['Gmag_syn', 'G_BPmag_syn', 'G_RPmag_syn'])
            else:
                raise ValueError('please give sample_obs while initial MockCMD()')
            g_med_err, bp_med_err, rp_med_err = e.estimate_med_photoerr(sample_syn=sample_syn)
            g_syn, bp_syn, rp_syn = e.syn_sample_photoerr(sample_syn=sample_syn)
            sample_syn[self.bands[0] + '_syn'], sample_syn[self.bands[1] + '_syn'], sample_syn[
                self.bands[2] + '_syn'] = (
                g_syn, bp_syn, rp_syn)
            sample_syn[self.bands[0] + '_err_syn'], sample_syn[self.bands[1] + '_err_syn'], sample_syn[
                self.bands[2] + '_err_syn'] = (
                g_med_err, bp_med_err, rp_med_err)

        # TODO: method 2, fit mag_magerr relation in sample_obs for each cluster directly
        # wait for developing
        '''
        if 'bands' in kwargs: # sample_obs have DIFFERENT band name/order with sample_syn
            print('NOTE! PLEASE ENTER THE OBSERVATION BAND ORDER ALIGNED WITH %s'%(self.bands))
            bands = kwargs['bands']
        bands = self.bands # sample_obs have SAME band name with sample_syn
        if 'bands_err' in kwargs:
            bands_err = kwargs['bands_err']
        bands_err = ['%s_err'%band for band in bands]
        # fit mag_magerr relation for each band
        for band,band_err in zip(bands,bands_err):
            x, y = sample_obs[band], sample_obs[band_err]
            mask = ~np.isnan(x) & ~np.isnan(y)
            x, y = x[mask], y[mask]
            mag_magerr = np.polyfit(x,y,3)
        '''
        end_time = time.time()
        run_time = end_time - start_time
        logging.info(f"time estimate_syn_photoerror() : {run_time:.4f} s")
        return sample_syn

    def mock_stars(self, theta, n_stars, step, isochrone=None):
        self.n_syn = n_stars

        start_time = time.time()
        logage, mh, fb, dm = theta  # Av, mag_min, mag_max not included yet!
        logage_step, mh_step = step
        # step 1: logage, m_h -> isochrone [mass, G, BP, RP]
        if isochrone:
            isochrone = pd.DataFrame(isochrone)
        else:
            isochrone = self.get_isochrone(
                logage=logage, mh=mh, dm=dm, logage_step=logage_step, mh_step=mh_step)

        # step 2: sample isochrone -> n_stars [ mass x [_pri, _sec], self.bands x [_pri, _sec, _syn]
        # single stars + binaries
        sample_syn = self.sample_imf(fb, isochrone, n_stars)
        # add dm & Av
        for _ in self.bands:
            sample_syn[_ + '_syn'] += dm
        # sample -> [ mass x [_pri, _sec], self.bands x [_pri, _sec, _syn]

        '''
        # draw CMD for checking
        c_2, m_2 = MockCMD.extract_cmd(sample_syn, band_a='Gmag_syn', band_b='G_BPmag_syn', band_c='G_RPmag_syn')
        fig,ax = plt.subplots(figsize=(5,5))
        ax.scatter(c_2,m_2,s=2,label='<-- add dm & sampleIMF -- isochrone')
        ax.set_ylabel('G (mag)')
        ax.set_xlabel('BP-RP (mag)')
        ax.invert_yaxis()
        ax.set_title('isochrone --> sampleIMF --> add dm')
        plt.show()
        '''

        # step 3: photometric uncertainties
        # interpolate & scale Edr3LogMagUncertainty
        # add uncertainties
        # sample -> [ mass x [_pri, _sec], self.bands x [_pri, _sec, _syn, _err_syn]
        sample_syn = self.estimate_syn_photerror(sample_syn)
        end_time = time.time()
        run_time = end_time - start_time
        logging.info(f"time mock_starts() : {run_time:.4f} s")
        return sample_syn  # sample

    @staticmethod
    def extract_cmd(sample, band_a, band_b, band_c):
        m = sample[band_a]
        c = sample[band_b] - sample[band_c]
        return c, m

    # @staticmethod
    # def draw_CMD(c,m):

    @staticmethod
    def hist2d_norm(c, m, c_grid=(0, 3, 0.2), m_grid=(6, 16, 0.1)):  # def hist2d(*sample.T,...):
        # adaptive grid wait for developing
        # define grid edges
        cstart, cend, cstep = c_grid
        mstart, mend, mstep = m_grid
        c_bin = np.arange(cstart, cend, cstep)
        m_bin = np.arange(mstart, mend, mstep)
        h, x_edges, y_edges = np.histogram2d(c, m, bins=(c_bin, m_bin))
        # h = h / np.sum(h)
        return h, x_edges, y_edges

    def eval_lnlikelihood(self, c_obs, m_obs, c_syn, m_syn, method="hist2hist", c_grid=(0, 3, 0.2),
                          m_grid=(6, 16, 0.1)):
        """Evaluate lnlikelihood.

        Parameters
        ----------
        c_obs
        m_obs
        c_syn
        m_syn
        c_grid
        m_grid
        method : str, optinal
            Method to calculate lnlikelihood. "hist2hist", "hist2point"
        """
        start_time = time.time()
        h_obs, _, _ = SynStars.hist2d_norm(c=c_obs, m=m_obs, c_grid=c_grid, m_grid=m_grid)
        h_syn, _, _ = SynStars.hist2d_norm(c=c_syn, m=m_syn, c_grid=c_grid, m_grid=m_grid)
        # non_zero_idx = np.where(h_obs > 0) # get indices of non-zero bins in h_obs
        # chi2 = np.sum(np.square(h_obs[non_zero_idx] - h_syn[non_zero_idx]) / h_obs[non_zero_idx])
        # chi2 = np.sum( np.square(h_obs - h_syn) / np.sqrt((h_obs+1) * (h_syn+1)) )
        if method == "hist2hist":
            h_syn = h_syn / (self.n_syn / self.n_obs)
            lnlike = -0.5 * np.sum(np.square(h_obs - h_syn) / (h_obs + h_syn + 1))
        elif method == "hist2point":
            epsilon = 1e-20
            h_syn = h_syn / np.sum(h_syn)
            h_syn = h_syn + epsilon
            h_syn = h_syn / np.sum(h_syn)
            lnlike = np.sum(h_obs * np.log10(h_syn))
        end_time = time.time()
        run_time = end_time - start_time
        logging.info(f"time eval_lnlikelihood() : {run_time:.4f} s")
        return lnlike


def main():
    name = 'Melotte_22'
    isochrones_dir = '/home/shenyueyue/Projects/Cluster/data/isocForMockCMD/'
    usecols = ['Gmag', 'G_BPmag', 'G_RPmag', 'phot_g_n_obs', 'phot_bp_n_obs', 'phot_rp_n_obs']
    sample_obs = pd.read_csv(
        "/home/shenyueyue/Projects/Cluster/data/Cantat-Gaudin_2020/%s.csv" % name, usecols=usecols)
    sample_obs = sample_obs.dropna().reset_index(drop=True)

    # theta = (6.83057773,-0.69887683,0.65960583,8.56889242)
    # theta = (7.14912235, -0.0367144, 0.5131032, 10.09228475)
    step = (0.05, 0.1)
    theta = (7.89, 0.032, 0.35, 5.55)
    # theta = (8.89, 0.032, 0.35, 5.55)
    n_stars = 100000

    m = SynStars(sample_obs=sample_obs, isochrones_dir=isochrones_dir)
    sample_syn = m.mock_stars(theta, n_stars, step)
    c_syn, m_syn = SynStars.extract_cmd(sample_syn, band_a='Gmag_syn', band_b='G_BPmag_syn', band_c='G_RPmag_syn')
    c_obs, m_obs = SynStars.extract_cmd(sample_obs, band_a='Gmag', band_b='G_BPmag', band_c='G_RPmag')
    lnlikelihood = m.eval_lnlikelihood(c_obs, m_obs, c_syn, m_syn, method="hist2point")
    print(lnlikelihood)

    '''
    # draw CMD for checking
    c_label='BP-RP (mag)'
    m_label='G (mag)'
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(8,4))
    ax[0].set_title('after estimate_syn_photoerror')
    ax[0].scatter(c_syn,m_syn,s=2)
    ax[0].invert_yaxis()
    ax[0].set_xlabel(c_label)
    ax[0].set_ylabel(m_label)
    # ax[0].set_xlim(min(c_obs),max(c_obs))
    # ax[0].set_ylim(max(m_obs),min(m_obs))

    ax[1].set_title('obs CMD')
    ax[1].scatter(c_obs,m_obs,s=2,label='obs')
    ax[1].invert_yaxis()
    ax[1].set_xlabel(c_label)
    ax[1].set_ylabel(m_label)
    plt.show()
    '''
    # fig,ax = plt.subplots(figsize=(5,5))
    # ax.scatter(c_syn, m_syn, s=2, c='orange', label='syn data')
    # ax.scatter(c_obs, m_obs, s=2, c='green', label='obs data')
    # ax.invert_yaxis()
    # ax.legend()
    # plt.show()


if __name__ == "__main__":
    main()
