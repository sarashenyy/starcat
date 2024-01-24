import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

from . import config, IMF, CMD
from .isoc import Isoc
from .logger import log_time


class SynStars(object):
    imf: IMF

    def __init__(self, model, photsys, imf, binmethod, photerr):
        """
        Synthetic cluster samples.

        Parameters
        ----------
        model : str
            'parsec' or 'MIST'
        photsys : str
            'CSST' or 'daiaDR3'
        imf : starcat.IMF
        binmethod :
            subclass of starcat.BinMethod: BinMS(), BinSimple()
        photerr :
            subclass of starcat.Photerr: GaiaEDR3()
        """
        self.imf = imf
        # self.n_stars = n_stars
        self.model = model
        self.photsys = photsys
        self.binmethod = binmethod
        self.photerr = photerr

        source = config.config[self.model][self.photsys]
        self.bands = source['bands']
        self.band_max_obs = source['band_max']
        self.band_max_syn = [x + 0.5 for x in source['band_max']]
        self.mini = source['mini']
        self.ext_coefs = source['extinction_coefs']
        self.mag = source['mag']
        self.color = source['color']

    @log_time
    def __call__(self, theta, n_stars, variable_type_isoc, test=False, figure=False, **kwargs):
        """
        Make synthetic cluster sample, considering binary method and photmetry error.
        Need to instantiate Isoc()(optional), sunclass of BinMethod and subclass of Photerr first.
        BinMethod : BinMS(), BinSimple()
        Photerr : GaiaEDR3()
        Isoc(Parsec()) / Isoc(MIST()) : optinal

        Parameters
        ----------
        theta : tuple
            logage, mh, dm, Av, fb
        n_stars : int
        step : tuple
            logage_step, mh_step
        isoc :
            starcat.Isoc() or pd.DataFRame, optional
            - starcat.Isoc() : Isoc(Parsec()), Isoc(MIST()). Default is None.
            - pd.DataFrame : isoc [phase, mini, [bands]]
        *kwargs :
            logage_step
            mh_step
        """
        logage, mh, dm, Av, fb, alpha = theta
        logage_step = kwargs.get('logage_step')
        mh_step = kwargs.get('mh_step')
        # !step 1: logage, mh ==> isoc [phase, mini, [bands]]
        if isinstance(variable_type_isoc, pd.DataFrame):
            isoc = variable_type_isoc
        elif isinstance(variable_type_isoc, Isoc):
            isoc = variable_type_isoc.get_isoc(
                self.photsys, logage=logage, mh=mh, logage_step=logage_step, mh_step=mh_step
            )
            if isoc is False:  # get_isoc() raise Error
                return False
        else:
            print('Please input an variable_type_isoc of type pd.DataFrame or starcat.Isoc.')

        # !step 2: add distance modulus and Av, make observed iso
        isoc_new = self.get_observe_isoc(isoc, dm, Av)

        if isoc_new is False:  # isoc_new cannot be observed
            return False

        # ?inspired by batch rejection sampling
        samples = pd.DataFrame()
        accepted = 0
        # batch_size = int(n_stars * 10)
        # runtime test
        if self.photsys == 'CSST':  # and len(self.bands) != 2
            best_rate = 1.2  # if discard only when all bands ar below magnitude limit
        else:  # or len(self.bands) == 2
            best_rate = 2
        batch_size = int(n_stars * best_rate)  # test results show that *1.2 can maximize the use of synthetic
        test_sample_time = 0
        total_size = batch_size

        while accepted < n_stars:
            # !step 3: sample isochrone with specified Binary Method
            #         ==> n_stars [ mass x [_pri, _sec], bands x [_pri, _sec, _syn]
            sample_syn = self.sample_stars(isoc_new, batch_size, fb, alpha=alpha)

            # !step 4: add photometry error for synthetic sample
            sample_syn = self.photerr.add_syn_photerr(sample_syn)

            # !step 5: (deleted!) 1. discard nan&inf values primarily due to failed interpolate
            # !        2. 只有在所有波段都暗于极限星等时才丢弃！即只有当一颗星在所有波段都不可见时才丢弃，只要这颗星在任一波段可见，则保留。
            # !           意味着最终返回的 samples[band] 中包含暗于该波段极限星等的值
            # # 1. discard nan&inf values primarily due to failed interpolate
            # columns_to_check = self.bands
            # sample_syn = sample_syn.dropna(subset=columns_to_check, how='any').reset_index(drop=True)
            # for column in columns_to_check:
            #     sample_syn = sample_syn[~np.isinf(sample_syn[column])]
            # # 2. discard sample_syn[mag] > band_max
            # if len(self.mag) == 1:
            #     sample_syn = sample_syn[sample_syn[self.mag[0]] <= self.band_max_obs[0]]
            # elif len(self.mag) > 1:
            #     for mag_col, band_max_val in zip(self.mag, self.band_max_obs):
            #         sample_syn = sample_syn[sample_syn[mag_col] <= band_max_val]

            if self.photsys == 'gaiaDR3' or len(self.bands) == 2:  # or len(self.bands) == 2
                # condition为只要有一个波段暗于极限星等，就把它丢弃
                condition = sample_syn[self.bands[0]] >= self.band_max_obs[0]
                for b, b_max in zip(self.bands[1:], self.band_max_obs[1:]):
                    cond = sample_syn[b] >= b_max
                    condition = condition | cond

            else:  # self.photsys == 'CSST'
                # condition为所有波段都暗于极限星等的星，将之丢弃
                condition = sample_syn[self.bands[0]] > self.band_max_obs[0]
                for b, b_max in zip(self.bands[1:], self.band_max_obs[1:]):
                    cond = sample_syn[b] > b_max
                    condition = condition & cond

            sample_syn = sample_syn[~condition].reset_index(drop=True)
            samples = pd.concat([samples, sample_syn], ignore_index=True)
            accepted += len(sample_syn)

            # dynamically adjusting batch_size
            if self.photsys == "gaiaDR3" or len(self.bands) == 2:
                if accepted < n_stars:
                    # remain = n_stars - accepted
                    # batch_size = int(remain * best_rate)
                    total_size += batch_size
            else:  # self.photsys == 'CSST'
                if accepted < n_stars:
                    remain = n_stars - accepted
                    batch_size = int(remain * best_rate)
                    total_size += batch_size

            test_sample_time += 1
            # rejection_rate = 1 - len(sample_syn) / batch_size
            # if rejection_rate > 0.2:
            #     batch_size = int(batch_size * 1.2)
            # else:
            #     batch_size = int(batch_size * 0.8)

            # runtime test

        samples = samples.iloc[:n_stars]
        # return samples
        # runtime test
        accepted_rate = accepted / total_size

        # test the process
        if figure is True:
            def _visualize_CSST_():
                fig = plt.figure(figsize=(10, 7.5))
                gs1 = GridSpec(2, 3, figure=fig, wspace=0.4, hspace=0.3)
                for i in range(len(self.bands) - 1):
                    ax = fig.add_subplot(gs1[int(i / 3), int(i % 3)])
                    c = isoc[self.bands[i]] - isoc[self.bands[i + 1]]
                    m = isoc[self.bands[i + 1]]
                    ax.plot(c, m, '-o', markersize=3)
                    ax.invert_yaxis()
                    ax.set_xlabel(f'{self.bands[i]}-{self.bands[i + 1]}')
                    ax.set_ylabel(f'{self.bands[i + 1]}')
                fig.suptitle(f'log(age)={logage}, [M/H]={mh}')
                fig.subplots_adjust(top=0.93)
                fig.show()

                fig = plt.figure(figsize=(10.5, 11))
                gs2 = GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.2)
                for i in range(len(self.bands) - 1):
                    _bin = samples['mass_sec'].notna()
                    ax = fig.add_subplot(gs2[int(i / 3), int(i % 3)])
                    _c = isoc_new[self.bands[i]] - isoc_new[self.bands[i + 1]]
                    _m = isoc_new[self.bands[i + 1]]
                    x = samples[self.bands[i]] - samples[self.bands[i + 1]]
                    y = samples[self.bands[i + 1]]
                    _uns = y > self.band_max_obs[i + 1]
                    ax.plot(_c, _m, '-', c='k', linewidth=0.5)
                    # stars that can be seen
                    ax.scatter(x[_bin & ~_uns], y[_bin & ~_uns], label='binary', s=3)
                    ax.scatter(x[~_bin & ~_uns], y[~_bin & ~_uns], label='single', s=3)
                    # stars that can NOT be seen
                    ax.scatter(x[_bin & _uns], y[_bin & _uns], color='grey', s=3)
                    ax.scatter(x[~_bin & _uns], y[~_bin & _uns], color='grey', s=3)
                    ax.text(0.1, 0.93, f'{len(x[~_uns])}/{int(n_stars)}', transform=ax.transAxes)
                    ax.set_ylim(min(y) - 0.5, max(y) + 0.5)
                    ax.set_xlim(-3, 5)
                    ax.invert_yaxis()
                    ax.set_xlabel(f'{self.bands[i]}-{self.bands[i + 1]}')
                    ax.set_ylabel(f'{self.bands[i + 1]}')
                fig.suptitle(f'DM={dm}, Av={Av}, fb={fb}')
                fig.subplots_adjust(top=0.95)
                fig.show()

            def _visualize_GAIA_():
                fig = plt.figure(figsize=(13, 5))
                gs = GridSpec(1, 3, figure=fig, hspace=0.15)
                c = samples['BP'] - samples['RP']
                m = samples['G']
                c_noe = (samples['BP'] - samples['BP_err']) - (samples['RP'] - samples['RP_err'])
                m_noe = samples['G'] - samples['G_err']
                bin = samples['mass_sec'].notna()

                ax3 = fig.add_subplot(gs[0, 2])
                ax3.scatter(c[bin], m[bin], label='binary', s=3)
                ax3.scatter(c[~bin], m[~bin], label='single', s=3)
                ax3.set_xlabel('BP-RP')
                ax3.invert_yaxis()

                ax2 = fig.add_subplot(gs[0, 1])
                ax2.scatter(c_noe[bin], m_noe[bin], label='binary', s=3)
                ax2.scatter(c_noe[~bin], m_noe[~bin], label='single', s=3)
                # ax2.legend()
                ax2.set_xlabel('BP-RP')
                ax2.invert_yaxis()

                ax1 = fig.add_subplot(gs[0, 0])
                ax1.plot(isoc_new['BP'] - isoc_new['RP'], isoc_new['G'], '-o', markersize=3)
                ax1.set_xlabel('BP-RP')
                ax1.set_ylabel('G')
                ax1.invert_yaxis()
                ax1.set_ylim(ax3.get_ylim())
                ax1.set_xlim(ax3.get_xlim())

                fig.suptitle(f'logage={logage},[M/H]={mh},DM={dm}, Av={Av}, fb={fb}')
                fig.subplots_adjust(top=0.92)
                fig.show()

            if self.photsys == 'CSST':
                _visualize_CSST_()
            elif self.photsys == 'gaiaDR3':
                _visualize_GAIA_()

        if test is True:
            return samples, accepted_rate, total_size, test_sample_time, isoc, isoc_new

        else:
            return samples

    def delta_color_samples(self, theta, n_stars, variable_type_isoc, **kwargs):
        """
        Make synthetic cluster sample, considering binary method and photmetry error.
        Need to instantiate Isoc()(optional), sunclass of BinMethod and subclass of Photerr first.
        BinMethod : BinMS(), BinSimple()
        Photerr : GaiaEDR3()
        Isoc(Parsec()) / Isoc(MIST()) : optinal

        Parameters
        ----------
        theta : tuple
            logage, mh, dm, Av, fb
        n_stars : int
        step : tuple
            logage_step, mh_step
        isoc :
            starcat.Isoc() or pd.DataFRame, optional
            - starcat.Isoc() : Isoc(Parsec()), Isoc(MIST()). Default is None.
            - pd.DataFrame : isoc [phase, mini, [bands]]
        *kwargs :
            logage_step
            mh_step
        """
        logage, mh, dm, Av, fb, alpha = theta
        logage_step = kwargs.get('logage_step')
        mh_step = kwargs.get('mh_step')
        # !step 1: logage, mh ==> isoc [phase, mini, [bands]]
        if isinstance(variable_type_isoc, pd.DataFrame):
            isoc = variable_type_isoc
        elif isinstance(variable_type_isoc, Isoc):
            isoc = variable_type_isoc.get_isoc(
                self.photsys, logage=logage, mh=mh, logage_step=logage_step, mh_step=mh_step
            )
            if isoc is False:  # get_isoc() raise Error
                return False
        else:
            print('Please input an variable_type_isoc of type pd.DataFrame or starcat.Isoc.')

        # !step 2: add distance modulus and Av, make observed iso
        isoc_new = self.get_observe_isoc(isoc, dm, Av)

        if isoc_new is False:  # isoc_new cannot be observed
            return False

        # ?inspired by batch rejection sampling
        samples = pd.DataFrame()
        accepted = 0
        # batch_size = int(n_stars * 10)
        # runtime test
        if self.photsys == 'CSST':  # and len(self.bands) != 2
            best_rate = 1.2  # if discard only when all bands ar below magnitude limit
        else:  # or len(self.bands) == 2
            best_rate = 2
        batch_size = int(n_stars * best_rate)  # test results show that *1.2 can maximize the use of synthetic
        test_sample_time = 0
        total_size = batch_size

        while accepted < n_stars:
            # !step 3: sample isochrone with specified Binary Method
            #         ==> n_stars [ mass x [_pri, _sec], bands x [_pri, _sec, _syn]
            sample_syn = self.sample_stars(isoc_new, batch_size, fb, alpha=alpha)

            # !step 4: add photometry error for synthetic sample
            sample_syn = self.photerr.add_syn_photerr(sample_syn)

            # !step 5: (deleted!) 1. discard nan&inf values primarily due to failed interpolate
            # !        2. 只有在所有波段都暗于极限星等时才丢弃！即只有当一颗星在所有波段都不可见时才丢弃，只要这颗星在任一波段可见，则保留。
            # !           意味着最终返回的 samples[band] 中包含暗于该波段极限星等的值
            # # 1. discard nan&inf values primarily due to failed interpolate
            # columns_to_check = self.bands
            # sample_syn = sample_syn.dropna(subset=columns_to_check, how='any').reset_index(drop=True)
            # for column in columns_to_check:
            #     sample_syn = sample_syn[~np.isinf(sample_syn[column])]
            # # 2. discard sample_syn[mag] > band_max
            # if len(self.mag) == 1:
            #     sample_syn = sample_syn[sample_syn[self.mag[0]] <= self.band_max_obs[0]]
            # elif len(self.mag) > 1:
            #     for mag_col, band_max_val in zip(self.mag, self.band_max_obs):
            #         sample_syn = sample_syn[sample_syn[mag_col] <= band_max_val]

            if self.photsys == 'gaiaDR3' or len(self.bands) == 2:  # or len(self.bands) == 2
                # condition为只要有一个波段暗于极限星等，就把它丢弃
                condition = sample_syn[self.bands[0]] >= self.band_max_obs[0]
                for b, b_max in zip(self.bands[1:], self.band_max_obs[1:]):
                    cond = sample_syn[b] >= b_max
                    condition = condition | cond

            else:  # self.photsys == 'CSST'
                # condition为所有波段都暗于极限星等的星，将之丢弃
                condition = sample_syn[self.bands[0]] > self.band_max_obs[0]
                for b, b_max in zip(self.bands[1:], self.band_max_obs[1:]):
                    cond = sample_syn[b] > b_max
                    condition = condition & cond

            sample_syn = sample_syn[~condition].reset_index(drop=True)
            samples = pd.concat([samples, sample_syn], ignore_index=True)
            accepted += len(sample_syn)

            # dynamically adjusting batch_size
            if self.photsys == "gaiaDR3" or len(self.bands) == 2:
                if accepted < n_stars:
                    # remain = n_stars - accepted
                    # batch_size = int(remain * best_rate)
                    total_size += batch_size
            else:  # self.photsys == 'CSST'
                if accepted < n_stars:
                    remain = n_stars - accepted
                    batch_size = int(remain * best_rate)
                    total_size += batch_size

            test_sample_time += 1
            # rejection_rate = 1 - len(sample_syn) / batch_size
            # if rejection_rate > 0.2:
            #     batch_size = int(batch_size * 1.2)
            # else:
            #     batch_size = int(batch_size * 0.8)

            # runtime test

        samples = samples.iloc[:n_stars]
        # return samples
        # runtime test
        accepted_rate = accepted / total_size

        c_syn, m_syn = CMD.extract_cmd(samples, self.model, self.photsys, True)
        isoc_c = (isoc_new[self.color[0][0]] - isoc_new[self.color[0][1]]).values.ravel()
        isoc_m = isoc_new[self.mag].values.ravel()
        c = (samples[self.color[0][0]] - samples[self.color[0][1]]).values.ravel()
        m = samples[self.mag].values.ravel()

        isoc_line = interp1d(x=isoc_m, y=isoc_c, fill_value='extrapolate')
        temp_c = isoc_line(x=m)
        delta_c = c - temp_c
        samples['delta_c'] = delta_c

        return samples

    def define_mass(self, isoc):
        """

        Parameters
        ----------
        isoc : pd.DataFrame
        dm : float

        """
        mass_max = max(isoc[self.mini])
        # if len(self.mag) == 1:
        #     try:
        #         mass_min = min(
        #             isoc[(isoc[self.mag[0]]) <= self.band_max_syn[0]][self.mini]
        #         )
        #     except ValueError:
        #         print('Do not have any stars brighter than the mag range!!')

        # elif len(self.mag) > 1:
        #     mass_min = min(
        #         isoc[(isoc[self.mag[0]]) <= self.band_max_syn[0]][self.mini]
        #     )
        #     for i in range(len(self.mag)):
        #         aux_min = min(
        #             isoc[(isoc[self.mag[i]]) <= self.band_max_syn[i]][self.mini]
        #         )
        #         if aux_min < mass_min:
        #             mass_min = aux_min

        aux_list = []
        for i in range(len(self.bands)):
            # synthetic Mini range is slightly larger than the observed for the consideration of binary and photerror
            condition = isoc[self.bands[i]] <= self.band_max_syn[i]
            filtered_isoc = isoc[condition]

            if not filtered_isoc.empty:
                aux_min = min(filtered_isoc[self.mini])
                aux_list.append(aux_min)
        mass_min = min(aux_list)
        # mass_min = 0.1

        return mass_min, mass_max

    def sample_stars(self, isoc, n_stars, fb, alpha=None):
        """
        Create sample of synthetic stars with specified binary method.

        Parameters
        ----------
        alpha : imf slope
        isoc : pd.DataFrame
        n_stars : int
        fb : float

        Returns
        -------
        pd.DataFrame :
            sample_syn ==> [ mass x [_pri, _sec], bands x [_pri, _sec, _syn]
        """
        # define mass range
        mass_min, mass_max = self.define_mass(isoc=isoc)
        # create synthetic sample of length n_stars
        sample_syn = pd.DataFrame(np.zeros((n_stars, 1)), columns=['mass_pri'])
        sample_syn['mass_pri'] = self.imf.sample(n_stars=n_stars, mass_min=mass_min,
                                                 mass_max=mass_max, alpha=alpha)
        # using specified binary method, see detail in binary.py
        sample_syn = self.binmethod.add_binary(
            fb, n_stars, sample_syn, isoc, self.imf, self.model, self.photsys
        )
        return sample_syn

    def get_observe_isoc(self, isoc, dm, Av):
        columns = isoc.columns
        isoc_new = pd.DataFrame(columns=columns)
        col_notin_bands = list(set(columns) - set(self.bands))
        isoc_new[col_notin_bands] = isoc[col_notin_bands]
        for _ in range(len(self.bands)):
            # get extinction coeficients
            l, w, c = self.ext_coefs[_]
            #    sample_syn[_] += dm
            isoc_new[self.bands[_]] = isoc[self.bands[_]] + dm + c * Av

            # if np.where(isoc_new[self.bands[_]] < self.band_max_obs)[0]
        return isoc_new

# def coefs_CSST(band):
#     """
#     From PARSEC CMD
#
#     Parameters
#     ----------
#     band: list
#
#     Returns
#     -------
#     λeff (Å), ωeff (Å), Aλ/AV
#     """
#     sys_param = {'NUV': [2887.74, 609, 1.88462],
#                  'u': [3610.40, 759, 1.55299],
#                  'g': [4811.96, 1357, 1.19715],
#                  'r': [6185.81, 1435, 0.86630],
#                  'i': [7641.61, 1536, 0.66204],
#                  'z': [9043.96, 1108, 0.47508],
#                  'y': [9660.53, 633, 0.42710]}
#
#     return sys_param[band]
