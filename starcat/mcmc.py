import logging
import time
import traceback
from multiprocessing import Pool

import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mock_cmd import MockCMD

plt.style.use("default")

# logfile in total.log

isochrones_dir = '/home/shenyueyue/Projects/Cluster/data/isocForMockCMD/'


def download_isochrone(isochrones_dir, logage_grid, mh_grid, dm, n_jobs=10):
    astart, aend, astep = logage_grid
    mstart, mend, mstep = mh_grid
    abin = np.arange(astart, aend, astep)
    mbin = np.arange(mstart, mend, mstep)
    logage_mh = []
    for a in abin:
        for m in mbin:
            logage_mh.append([a, m])
    logging.info(f"dowanload in total : {len(logage_mh)} isochrones")

    # nested function, access variable in parent function
    def get_isochrone_wrapper(logage, mh):
        m = MockCMD(isochrones_dir=isochrones_dir)
        m.get_isochrone(logage=logage, mh=mh, dm=dm, logage_step=astep, mh_step=mstep)
        logging.info(f"get_iso : logage = {logage}, mh = {mh}")

    # parallel excution
    Parallel(n_jobs=n_jobs)(
        delayed(get_isochrone_wrapper)(logage, mh) for logage, mh in logage_mh
    )


def lnlike_distribution(sample_obs, logage_grid, mh_grid, n_jobs=10):
    astart, aend, astep = logage_grid
    mstart, mend, mstep = mh_grid
    step = (astep, mstep)
    abin = np.arange(astart, aend, astep)
    mbin = np.arange(mstart, mend, mstep)
    n_stars = len(sample_obs)
    logage_mh = []
    for a in abin:
        for m in mbin:
            logage_mh.append([a, m])
    print(f"calculate in total : {len(logage_mh)} lnlike values")

    # nested function, access variable in parent function
    def lnlike_wrapper(theta_part):
        lnlikelihood = lnlike(theta_part, n_stars, step, sample_obs)
        return lnlikelihood

    # parallel excution
    results = Parallel(n_jobs=n_jobs)(
        delayed(lnlike_wrapper)(theta_part) for theta_part in logage_mh
    )
    # Create DataFrame
    df = pd.DataFrame(logage_mh, columns=['logage', 'mh'])
    df['lnlike'] = results
    return df


def lnlike(theta_part, n_stars, step, sample_obs, method="hist2hist"):
    start_time = time.time()
    # logage, mh, fb, dm = theta
    fb, dm = 0.35, 5.55
    logage, mh = theta_part
    theta = logage, mh, fb, dm
    try:
        if (logage > 10.0) or (logage < 6.6) or (mh < -0.9) or (mh > 0.7) or (fb < 0.05) or (fb > 1) or (dm < 2) or (
                dm > 20):
            return -np.inf
        m = MockCMD(sample_obs=sample_obs, isochrones_dir=isochrones_dir)
        sample_syn = m.mock_stars(theta, n_stars, step)
        c_syn, m_syn = MockCMD.extract_cmd(sample_syn, band_a='Gmag_syn', band_b='G_BPmag_syn', band_c='G_RPmag_syn')
        c_obs, m_obs = MockCMD.extract_cmd(sample_obs, band_a='Gmag', band_b='G_BPmag', band_c='G_RPmag')
        lnlikelihood = m.eval_lnlikelihood(c_obs, m_obs, c_syn, m_syn, method)
        end_time = time.time()
        run_time = end_time - start_time
        logging.info(f"time lnlike() : {run_time:.4f} s")
        return lnlikelihood

    except Exception as e:
        logging.error("lnlike:", lnlikelihood)
        logging.error("Error parameters: [%f, %f, %f, %f]" % (logage, mh, fb, dm))
        logging.error(f"Error encountered: {e}")
        traceback.print_exc()
        return -np.inf


def test_randomness(sample_obs, theta, n_stars, method="hist2hist", time=2000, step=(0.05, 0.1)):
    """Test the randomness of lnlikelihood.

    Parameters
    ----------
    method : str, optinal
        Method to calculate lnlikelihood. "hist2hist", "hist2point"
    """
    lnlike_list = []
    logage, mh, fb, dm = theta
    print(f"calculate lnlike values in total : {time} times")
    for i in range(time):
        lnlikelihood = lnlike(theta, n_stars, step, sample_obs, method)
        lnlike_list.append(lnlikelihood)
        if i % 100 == 0:
            print(i)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(lnlike_list, bins=50)
    ax.set_xlabel('lnlikelihood')
    ax.text(0.7, 0.95, f"mean: {np.mean(lnlike_list):.1f}", transform=ax.transAxes)
    ax.text(0.7, 0.90, f"std: {np.std(lnlike_list):.1f}", transform=ax.transAxes)
    ax.set_title(f"logage:{logage} [M/H]:{mh} fb:{fb} dm:{dm} nstars:{n_stars}")
    fig.show()
    return lnlike_list


# def test_randommess_n_stars():
#
#     # test the randomness of lnlikelihood() vs n_stars
#     n = [500, 942, 1000, 5000]
#     for n_stars in n:
#         print(f"test_randomness for n_stars={n_stars}")
#         test_randomness(sample_obs, theta1, n_stars)
#
#     # test the lnlike distribution in parameter space
#     logage_grid = (6.6, 10, 0.01)
#     mh_grid = (-0.9, 0.7, 0.01)
#     df = lnlike_distribution(sample_obs, logage_grid=logage_grid, mh_grid=mh_grid)
#     df.to_csv("/home/shenyueyue/Projects/Cluster/data/lnlike_distribution.csv",index=False)


def read_sample_obs():
    name = 'Melotte_22'
    usecols = ['Gmag', 'G_BPmag', 'G_RPmag', 'phot_g_n_obs', 'phot_bp_n_obs', 'phot_rp_n_obs']
    sample_obs = pd.read_csv("/home/shenyueyue/Projects/Cluster/data/Cantat-Gaudin_2020/%s.csv" % name,
                             usecols=usecols)
    sample_obs = sample_obs.dropna().reset_index(drop=True)
    return sample_obs


def test():
    sample_obs = read_sample_obs()
    # test the randomness of lnlikelihood() vs logage
    theta1 = (7.89, 0.032, 0.35, 5.55)  #
    theta2 = (8.89, 0.032, 0.35, 5.55)  #
    n_stars = 100000
    method = "hist2hist"
    test_randomness(sample_obs, theta1, n_stars, method)
    # test_randomness(sample_obs, theta2, n_stars, method)
    return


def main():
    # read smaple_obs
    name = 'Melotte_22'
    usecols = ['Gmag', 'G_BPmag', 'G_RPmag', 'phot_g_n_obs', 'phot_bp_n_obs', 'phot_rp_n_obs']
    sample_obs = pd.read_csv("/home/shenyueyue/Projects/Cluster/data/Cantat-Gaudin_2020/%s.csv" % name,
                             usecols=usecols)
    sample_obs = sample_obs.dropna().reset_index(drop=True)

    '''
    # download isochrone
    logage_grid = (6.6, 10, 0.01)
    mh_grid = (-0.9, 0.7, 0.01)
    dm = 5.55 
    download_isochrone(isochrones_dir=isochrones_dir, logage_grid=logage_grid, mh_grid=mh_grid, dm=dm)
    '''

    # MCMC
    n_stars = len(sample_obs)
    step = (0.05, 0.2)
    # parameter
    theta_part = np.array(
        [7.80, 0.03])  # after round theta = (8.00, 0.0)  Dias+,2021:8.116,0.032 Cantat-Gaudin+,2020:7.89
    scale = np.array([0.1, 0.1])
    ndim = 2

    # set up the MCMC sampler
    nwalkers = 100
    method = "hist2hist"
    # define the step sizes for each parameter
    # moves = [emcee.moves.StretchMove(a=step_sizes[i]) for i in range(ndim)]
    # create an array of initial positions with small random perturbations
    p0 = np.round((theta_part + scale * np.random.randn(nwalkers, ndim)), decimals=2)

    # parallelization
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(n_stars, step, sample_obs, method),
                                        pool=pool)  # , moves=moves
        # burn-in
        nburn = 100
        start_burn = time.time()
        pos, _, _ = sampler.run_mcmc(p0, nburn, progress=True)
        end_burn = time.time()
        time_burn = end_burn - start_burn
        print(f"burn-in time : {time_burn:.1f} seconds")
        sampler.reset()

        # run the MCMC sampler
        nsteps = 2000
        start_run = time.time()
        sampler.run_mcmc(pos, nsteps, progress=True)
        end_run = time.time()
        time_run = end_run - start_run
        print(f"run time : {time_run:.1f} seconds")

    samples = sampler.chain[:, :, :].reshape((-1, ndim))
    fig = corner.corner(samples,
                        labels=[r'log(age)', r'[M/H]', r'$f_b$', r'DM'],
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 12},
                        title_fmt='.2f')
    fig.savefig(f"/home/shenyueyue/Projects/Cluster/code/point_source/figure/mcmc_w{nwalkers}_b{nburn}_r{nsteps}.png",
                bbox_inches='tight')


def test_corner():
    ndim = 2
    nwalkers = 4

    p = np.random.randn(nwalkers, ndim)

    def log_prob_fn(p):
        return -0.5 * np.sum(p ** 2)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn)
    sampler.run_mcmc(p, 1000, progress=True)
    samples = sampler.chain[:, :, :].reshape((-1, ndim))
    fig = corner.corner(samples)
    return fig


if __name__ == '__main__':
    test()
