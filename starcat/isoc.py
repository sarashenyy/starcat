import os
import time
from abc import ABC, abstractmethod

import joblib
import pandas as pd
from berliner import CMD

from . import config
from .logger import log_time
from .widgets import round_to_step


class Isoc(object):
    """
    Isochrone

    """

    def __init__(self, model):
        """

        Parameters
        ----------
        model : IsocModel
        """
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
        parsec = Parsec()
        i = Isoc(Parsec)
        isoc = i.get_isoc
        ```
        """
        return self.model.get_isoc(photsyn=photsyn, **kwargs)

    def get_obsisoc(self, photsyn, **kwargs):
        return self.model.get_obsisoc(photsyn=photsyn, **kwargs)

    def bulk_load(self, photsyn, logage_grid, mh_grid, n_jobs=-1, **kwargs):
        """
        Bulk Laod isochrones. DISCARDED DUE TO BUGS!!

        Parameters
        ----------
        logage_grid
        mh_grid
        n_jobs : int, optional
            Default 20.
        photsyn
        kwargs : dict
            - logage_grid (tuple): start, end, step
            - mh_grid (tuple): start, end, step
        """
        self.model.bulk_load(photsyn=photsyn, logage_grid=logage_grid, mh_grid=mh_grid, n_jobs=n_jobs)


class IsocModel(ABC):
    """
    An abstract base class for isochrone model that defines the interface for its subclass.
    """

    def load_model(self, photsyn):
        pass

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

    def get_obsisoc(self, phoysyn, **kwargs):
        pass

    def bulk_load(self, photsyn, logage_grid, mh_grid, n_jobs, **kwargs):
        """
        Bulk Laod isochrones.

        Parameters
        ----------
        n_jobs
        photsyn
        logage_grid: tuple
            start, end, step
        mh_grid: tuple
            start, end, step
        """
        pass


class Parsec(IsocModel):
    """
    subclass for abstract base class Model()
    """

    def __init__(self, **kwargs):
        self.model = 'parsec'

        photsyn = kwargs.get('photsyn')
        if photsyn is not None:
            self.loaded_data = self.load_model(photsyn)
            self.photsyn = photsyn

    def load_model(self, photsyn):
        source = config.config[self.model][photsyn]
        isoc_dir = source['isoc_dir']
        folder_path = config.data_dir + isoc_dir
        # 获取文件夹中的文件列表
        file_list = os.listdir(folder_path)
        file_list = [file for file in file_list if file.endswith('.joblib')]
        # 用于存储加载的数据的字典
        loaded_data = {}
        # 批量读取文件并加载到内存中
        start = time.time()
        for i, file_name in enumerate(file_list):
            # if i % 100 == 0:
            #     print(i)
            file_path = os.path.join(folder_path, file_name)
            loaded_data[file_name] = joblib.load(file_path)
        end = time.time()
        print(f'load {len(file_list)} isochrones using {end - start:.4f}s')
        return loaded_data

    @log_time
    def get_isoc(self, photsyn, **kwargs):
        """
        Get isochrone from parsec model.

        Parameters
        ----------
        photsyn : str, optinal
            The synthetic photometry to use for isochrone. For example: "gaiaDR2". See config.toml for more options.
        **kwargs : dict
            - logage (float): logarithmic age
            - logage_step (float): step size of logage
            - mh (float): [M/H]
            - mh_step (float): step size of [M/H]
            - track_parsec: str, optional
                parsec version, default is 'parsec_CAF09_v1.2S', can also choose 'parsec_CAF09_v2.0'
        Returns
        -------
        pd.DataFrame: isochrone containing the evolutionary phase, initial mass and photometry bands.
        """
        # logage = round_to_step(kwargs.get('logage'), step=kwargs.get('logage_step'))
        # mh = round_to_step(kwargs.get('mh'), step=kwargs.get('mh_step'))
        logage_step = kwargs.get('logage_step')
        mh_step = kwargs.get('mh_step')
        if logage_step is None or logage_step == 0:
            logage = kwargs.get('logage')
        else:
            logage = round_to_step(kwargs.get('logage'), step=kwargs.get('logage_step'))
        if mh_step is None or mh_step == 0:
            mh = kwargs.get('mh')
        else:
            mh = round_to_step(kwargs.get('mh'), step=kwargs.get('mh_step'))

        if kwargs.get('track_parsec') is None:
            track_parsec = 'parsec_CAF09_v1.2S'
        else:
            track_parsec = kwargs.get('track_parsec')

        source = config.config[self.model][photsyn]
        bands_isoc = source['bands_isoc']
        bands = source['bands']
        mini = source['mini']
        # mass = source['mass']
        # label = source['label']
        # phase = source['phase']
        isoc_dir = source['isoc_dir']
        if mh == -0.:  # change -0.0 to +0.0
            mh = 0.
        isoc_path = config.data_dir + isoc_dir + f'age{logage:+.2f}_mh{mh:+.2f}.joblib'

        # EOFE_flag = True
        # if os.path.exists(isoc_path):
        #     try:
        #         isochrone = joblib.load(isoc_path)
        #         # change bands name
        #         # flag = False
        #         # for band_isoc, band in zip(bands_isoc, bands):
        #         #     if band_isoc in isochrone.columns:
        #         #         flag = True
        #         #         isochrone = isochrone.rename(columns={band_isoc: band})
        #         # if flag:
        #         #     joblib.dump(isochrone, isoc_path)
        #     except EOFError:
        #         EOFE_flag = False
        #
        # elif os.path.exists(isoc_path) is False or EOFE_flag is False:
        #     isochrone = pd.DataFrame([])
        #     max_attempt = 3
        #     attempt_time = 0
        #     while isochrone.empty and attempt_time <= max_attempt:
        #         c = CMD()
        #         isochrone = c.get_one_isochrone(
        #             logage=logage, z=None, mh=mh, photsys_file=photsyn, track_parsec=track_parsec
        #         )
        #         # do not truncate isochrone, drop the last point instead
        #         isochrone = isochrone[0:-1].to_pandas()
        #         joblib.dump(isochrone, isoc_path)
        #         attempt_time += 1
        #         # truncate isochrone, PMS~EAGB
        #         # ATTENTION! parsec use "label" to represent evolutionary phase, different from MIST("phase")
        #         # isochrone = isochrone[
        #         #     (isochrone['label'] >= min(label)) & (isochrone['label'] <= max(label))].to_pandas()
        #
        #         # add evolutionary phase info
        #         # for i, element in enumerate(label):
        #         #     index = np.where(isochrone['label'] == element)[0]
        #         #     isochrone.loc[index, 'phase'] = phase[i]
        #         # change bands name
        #         # rename_dict = dict(zip(bands_isoc, bands))
        #         # isochrone = isochrone.rename(columns=rename_dict)
        #         # save isochrone file
        #         # if not os.path.exists(isoc_path):

        # useful_columns = ['phase', mini, mass] + bands

        # if os.path.exists(isoc_path):
        #     isochrone = joblib.load(isoc_path)
        file_name = f'age{logage:+.2f}_mh{mh:+.2f}.joblib'
        if os.path.exists(isoc_path):
            isochrone = self.loaded_data.get(file_name)
        else:
            c = CMD()
            if photsyn == 'gaiaDR3':
                photsys_file = 'gaiaEDR3'
            else:
                photsys_file = photsyn
            try:
                isochrone = c.get_one_isochrone(
                    logage=logage, z=None, mh=mh, photsys_file=photsys_file, track_parsec=track_parsec
                )
            except:
                print(f'logage={logage}, [M/H]={mh}, photsys_file={photsys_file}')
                return False
            isochrone = isochrone[0:-1].to_pandas()
            joblib.dump(isochrone, isoc_path)

        # label: 0=PMS, 1=MS, 2=SGB, 3=RGB, (4,5,6)=different stages of CHEB
        # !! NOTICE MS for test ONLY!!
        # isochrone = isochrone[(isochrone['label'] >= 0) & (isochrone['label'] <= 1)]
        isochrone = isochrone[(isochrone['label'] >= 0) & (isochrone['label'] <= 6)]
        useful_columns = [mini] + bands_isoc
        try:
            isoc = isochrone[useful_columns]
            # ! begin to use the renamed isochorne (detail in config.toml: bands_isoc & bands)
            rename_dict = dict(zip(bands_isoc, bands))
            isoc = isoc.rename(columns=rename_dict)
            return isoc
        except UnboundLocalError:
            print(f'logage={logage}, [M/H]={mh} occurs UnboundLocal Error in getting isochrone.')
            return False

    def get_obsisoc(self, photsyn, **kwargs):
        source = config.config[self.model][photsyn]
        bands = source['bands']
        ext_coefs = source['extinction_coefs']

        dm = kwargs.get('dm')
        Av = kwargs.get('Av')
        logage_step = kwargs.get('logage_step')
        mh_step = kwargs.get('mh_step')
        logage = kwargs.get('logage')
        mh = kwargs.get('mh')

        abs_isoc = self.get_isoc(photsyn, logage=logage, mh=mh, logage_step=logage_step, mh_step=mh_step)
        columns = abs_isoc.columns
        isoc_new = pd.DataFrame(columns=columns)

        col_notin_bands = list(set(columns) - set(bands))
        isoc_new[col_notin_bands] = abs_isoc[col_notin_bands]

        for _ in range(len(bands)):
            # get extinction coeficients
            l, w, c = ext_coefs[_]
            #    sample_syn[_] += dm
            isoc_new[bands[_]] = abs_isoc[bands[_]] + dm + c * Av

            # if np.where(isoc_new[self.bands[_]] < self.band_max_obs)[0]

        return isoc_new

    def bulk_load(self, photsyn, logage_grid, mh_grid, n_jobs, **kwargs):
        # """
        # Bulk Laod isochrones.
        #
        # Parameters
        # ----------
        # n_jobs
        # photsyn
        # logage_grid: tuple
        #     start, end, step
        # mh_grid: tuple
        #     start, end, step
        # """
        # astart, aend, astep = logage_grid
        # mstart, mend, mstep = mh_grid
        # # NOTE: np.around(decimals=4) to cut np.arange() results
        # abin = np.around(np.arange(astart, aend, astep), decimals=4)
        # mbin = np.around(np.arange(mstart, mend, mstep), decimals=4)
        #
        # if kwargs.get('track_parsec') is None:
        #     track_parsec = 'parsec_CAF09_v1.2S'
        # else:
        #     track_parsec = kwargs.get('track_parsec')
        #
        # source = config.config[self.model][photsyn]
        # isoc_dir = source['isoc_dir']
        #
        # print(f'Total download {len(abin) * len(mbin)} isochrones: n(logAge)={len(abin)}, n([M/H])={len(mbin)}')
        # print(f'grid(logAge): {abin}')
        # print(f'grid([M/H]: {mbin}')
        # logage_mh = []
        # for a in abin:
        #     for m in mbin:
        #         logage_mh.append([a, m])
        #
        # # nested function, access variable in parent function
        # def bulk_load_wrapper(__logage, __mh):
        #     c = CMD()
        #     isoc = c.get_one_isochrone(
        #         logage=__logage, z=None, mh=__mh, photsys_file=photsyn, track_parsec=track_parsec
        #     )
        #     isoc = isoc[0:-1].to_pandas()
        #     isoc_path = config.data_dir + isoc_dir + f'age{__logage:+.2f}_mh{__mh:+.2f}.joblib'
        #     joblib.dump(isoc, isoc_path)
        #     return isoc
        #
        # # parallel excution
        # with joblib_progress('Downloading isochrones...', total=len(logage_mh)):
        #     results = Parallel(n_jobs=n_jobs)(
        #         delayed(bulk_load_wrapper)(_logage, _mh) for _logage, _mh in logage_mh
        #     )
        #
        # # check
        # for (logage, mh), result in zip(logage_mh, results):
        #     if result is False:
        #         print(f'Failed to retrieve isochrone for logage={logage}, [M/H]={mh}')
        # return results
        pass


class MIST(IsocModel):
    """
    subclass for abstract base class Model()
    """

    def __init__(self, **kwargs):
        self.model = 'mist'

        photsyn = kwargs.get('photsyn')
        if photsyn is not None:
            self.loaded_data = self.load_model(photsyn)
            self.photsyn = photsyn

    def load_model(self, photsyn):
        source = config.config[self.model][photsyn]
        isoc_dir = source['isoc_dir']
        folder_path = config.data_dir + isoc_dir
        # 获取文件夹中的文件列表
        file_list = os.listdir(folder_path)
        file_list = [file for file in file_list if file.endswith('.joblib')]
        # 用于存储加载的数据的字典
        loaded_data = {}
        # 批量读取文件并加载到内存中
        start = time.time()
        for i, file_name in enumerate(file_list):
            # if i % 100 == 0:
            #     print(i)
            file_path = os.path.join(folder_path, file_name)
            loaded_data[file_name] = joblib.load(file_path)
        end = time.time()
        print(f'load {len(file_list)} isochrones using {end - start:.4f}s')
        return loaded_data

    def get_isoc(self, photsyn, **kwargs):
        logage_step = kwargs.get('logage_step')
        mh_step = kwargs.get('mh_step')
        if logage_step is None or logage_step == 0:
            logage = kwargs.get('logage')
        else:
            logage = round_to_step(kwargs.get('logage'), step=kwargs.get('logage_step'))
        if mh_step is None or mh_step == 0:  # mh_step = 0.5 only!
            mh = kwargs.get('mh')
        else:
            mh = round_to_step(kwargs.get('mh'), step=kwargs.get('mh_step'))

        source = config.config[self.model][photsyn]
        bands_isoc = source['bands_isoc']
        bands = source['bands']
        mini = source['mini']
        # mass = source['mass']
        # label = source['label']
        # phase = source['phase']
        isoc_dir = source['isoc_dir']
        if mh == -0.:  # change -0.0 to +0.0
            mh = 0.
        isoc_path = config.data_dir + isoc_dir + f'age{logage:+.2f}_mh{mh:+.2f}.joblib'
        if os.path.exists(isoc_path):
            isochrone = joblib.load(isoc_path)
        else:
            print(f'please download logage={logage:+.2f}, [M/H]={mh:+.2f}')

        # phase: -1=PMS, 0=MS, 2=RGB, 3=CHeB
        isochrone = isochrone[(isochrone['phase'] >= -1) & (isochrone['phase'] <= 3)]
        useful_columns = [mini] + bands_isoc
        try:
            isoc = isochrone[useful_columns]
            # ! begin to use the renamed isochorne (detail in config.toml: bands_isoc & bands)
            rename_dict = dict(zip(bands_isoc, bands))
            isoc = isoc.rename(columns=rename_dict)
            return isoc
        except UnboundLocalError:
            print(f'logage={logage}, [M/H]={mh} occurs UnboundLocal Error in getting isochrone.')
            return False

    def bulk_load(self, photsyn, logage_grid, mh_grid, n_jobs, **kwargs):
        """
        Bulk Laod isochrones.

        Parameters
        ----------
        n_jobs
        photsyn
        logage_grid: tuple
            start, end, step
        mh_grid: tuple
            start, end, step
        """
        pass
