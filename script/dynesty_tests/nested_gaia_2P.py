import dynesty
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dynesty import plotting as dyplot

from starcat import (Isoc, Parsec, IMF,
                     BinMRD,
                     SynStars,
                     lnlike, Hist2Hist4CMD, Hist2Point4CMD, Individual)

plt.style.use('/Users/sara/PycharmProjects/starcat/data/mystyle.mplstyle')
file_path = '/Users/sara/PycharmProjects/starcat/data/Hunt23/NGC_3532.csv'


def loglikelihood(x, step, isoc_inst, likelihood_inst, synstars_inst, n_syn, observation):
    return lnlike(x, step,
                  isoc_inst, likelihood_inst, synstars_inst, n_syn,
                  observation, 'LG', 5,
                  logage=8., mh=0., dm=18.5, Av=0.5)


def prior_transform(u):
    """Transforms the uniform random variables `u ~ Unif[0., 1.)`
    to the parameters of interest."""
    x = u.copy()  # copy u
    x_range = [[0.2, 1.0], [1.6, 3.0]]
    for i, xrange in enumerate(x_range):
        x[i] = (1 - u[i]) * xrange[0] + u[i] * xrange[1]
    return x


if __name__ == "__main__":
    obs_sample = pd.read_csv(file_path, usecols=['Prob', 'Gmag', 'BPmag', 'RPmag', 'G_err', 'BP_err', 'RP_err'])

    Pobservation = obs_sample[obs_sample['Prob'] > 0.6]
    Pobservation = Pobservation[(Pobservation['BPmag'] - Pobservation['RPmag']) > -0.2]
    Pobservation = Pobservation[Pobservation['Gmag'] < 18.]

    c = Pobservation['BPmag'] - Pobservation['RPmag']
    sigma_c = np.sqrt((Pobservation['BP_err']) ** 2 + (Pobservation['RP_err']) ** 2)
    Pobservation = Pobservation[sigma_c < np.std(sigma_c) * 6]

    print(max(Pobservation['Gmag']), max(Pobservation['BPmag']), max(Pobservation['RPmag']))
    print(len(Pobservation))

    fig, ax = plt.subplots(figsize=(3, 4))
    ax.scatter(Pobservation['BPmag'] - Pobservation['RPmag'], Pobservation['Gmag'], s=3)
    ax.set_ylim(max(Pobservation['Gmag']) + 1, min(Pobservation['Gmag']) - 1)
    fig.show()

    photsys = 'gaiaDR3'
    model = 'parsec'
    imf = 'salpeter55'
    imf_inst = IMF(imf)
    parsec_inst = Parsec()
    # binmethod = BinSimple()
    binmethod = BinMRD()
    # med_obs=[200,20,20]
    # photerr = GaiaDR3(model, med_obs)
    photerr = Individual(model, photsys, Pobservation)
    isoc_inst = Isoc(parsec_inst)
    synstars_inst = SynStars(model, photsys,
                             imf_inst, binmethod, photerr)

    logage, mh, dm, Av, fb, alpha = 8.6, 0.0, 8.45, 0.18, 0.27, 2.35
    theta_args = fb, alpha
    theta = logage, mh, dm, Av, fb, alpha
    step = (0.05, 0.05)  # logage, mh
    n_stars = 1500
    n_syn = 100000
    bins = 50
    h2h_cmd_inst = Hist2Hist4CMD(model, photsys, bins)
    h2p_cmd_inst = Hist2Point4CMD(model, photsys, bins)

    samples = synstars_inst(theta, n_stars, isoc_inst)
    observation = samples
    observation = observation.rename(columns={'G': 'Gmag', 'BP': 'BPmag', 'RP': 'RPmag'})

    fig, ax = plt.subplots(figsize=(4, 5))
    bin = observation['mass_sec'].notna()
    c = observation['BPmag'] - observation['RPmag']
    m = observation['Gmag']

    ax.scatter(c[bin], m[bin], color='blue', s=5, alpha=0.6, label='binary')
    ax.scatter(c[~bin], m[~bin], color='orange', s=5, alpha=0.6, label='single')
    ax.set_ylim(max(m) + 0.2, min(m) - 0.5)
    ax.set_xlim(min(c) - 0.2, max(c) + 0.2)
    ax.legend()
    ax.set_title(f'logage={logage}, [M/H]={mh}, DM={dm}, \n'
                 f'Av={Av}, fb={fb}, alpha={alpha}', fontsize=12)
    ax.set_xlabel('BP - RP')
    ax.set_ylabel('G')

    fig.show()

    samples = synstars_inst(theta, n_syn, isoc_inst)
    h2h = lnlike(theta_args, step,
                 isoc_inst, h2h_cmd_inst, synstars_inst, n_syn,
                 observation, 'MW', 5,
                 logage=logage, mh=mh, dm=dm, Av=Av)
    h2p = lnlike(theta_args, step,
                 isoc_inst, h2p_cmd_inst, synstars_inst, n_syn,
                 observation, 'MW', 5,
                 logage=logage, mh=mh, dm=dm, Av=Av)
    print(h2h)
    print(h2p)

    fig, ax = plt.subplots(figsize=(4, 4))
    bin = samples['mass_sec'].notna()
    c = samples['BP'] - samples['RP']
    m = samples['G']
    c_obs = observation['BPmag'] - observation['RPmag']
    m_obs = observation['Gmag']

    ax.scatter(c[bin], m[bin], color='blue', s=5, alpha=0.6)
    ax.scatter(c[~bin], m[~bin], color='orange', s=5, alpha=0.6)
    ax.scatter(c_obs, m_obs, s=1, color='grey')
    ax.invert_yaxis()
    ax.set_xlim(min(c_obs) - 0.2, max(c_obs) + 0.2)
    ax.set_ylim(max(m_obs) + 0.5, min(m_obs) - 0.5)
    ax.set_title(f'logage={logage}, [M/H]={mh}, DM={dm}, \n'
                 f'Av={Av}, fb={fb}, alpha={alpha}', fontsize=12)
    ax.set_xlabel('BP - RP')
    ax.set_ylabel('G')
    ax.text(0.6, 0.6, f'H2H={h2h:.2f}', transform=ax.transAxes, fontsize=11)
    ax.text(0.6, 0.5, f'H2P={h2p:.2f}', transform=ax.transAxes, fontsize=11)

    fig.show()

    ndim = 2
    likelihood_inst = h2p_cmd_inst

    # sampler = dynesty.NestedSampler(loglike, ptform, ndim)
    # sampler.run_nested()
    # loglike_args = (step, isoc_inst, likelihood_inst, synstars_inst, n_syn, observation, 'LG', 5)
    # Set up the dynamic nested sampling run with multiprocessing
    with dynesty.pool.Pool(20, loglikelihood, prior_transform,
                           logl_args=(step, isoc_inst, likelihood_inst, synstars_inst, n_syn, observation)) as pool:
        sampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform, ndim, pool=pool,
                                        nlive=1000, bound='multi', sample='rwalk')
        sampler.run_nested(dlogz=0.1, checkpoint_file='H2P.2P.gaia.save')

    filename = 'H2P.2P.gaia.sampler'
    dir = '/Users/sara/PycharmProjects/starcat/script/dynesty_tests/'
    path = dir + filename
    joblib.dump(sampler, path)

    results = sampler.results

    label = ['$f_b$', '$\\alpha$']
    # summary (run) plot
    fig, axes = dyplot.runplot(results)
    fig.tight_layout()
    fig.show()

    # Trace Plots: generate a trace plot showing the evolution of particles
    # (and their marginal posterior distributions) in 1-D projections.
    # colored by importance weight
    # Highlight specific particle paths (shown above) to inspect the behavior of individual particles.
    # (These can be useful to qualitatively identify problematic behavior such as strongly correlated samples.)
    fig, axes = dyplot.traceplot(results, truths=theta_args,
                                 labels=label,
                                 truth_color='black', show_titles=True,
                                 title_kwargs={'fontsize': 24},
                                 trace_cmap='viridis', connect=True,
                                 connect_highlight=range(5)
                                 )  # fig=plt.subplots(6, 2, figsize=(15, 20))
    fig.tight_layout()
    # fig.subplots_adjust(wspace=0.1, hspace=0.8)
    fig.show()

    # Corner Points
    # kde=True: colored according to their estimated posterior mass
    # kde=False: colored according to raw importance weights
    fg, ax = dyplot.cornerpoints(results, cmap='plasma', truths=theta_args,
                                 truth_color='red',
                                 labels=label,
                                 kde=True)
    fg.show()

    #  Corner Plot
    fg, ax = dyplot.cornerplot(results, color='blue', truths=theta_args,
                               labels=label,
                               truth_color='black', show_titles=True,
                               max_n_ticks=3, quantiles=None)
    fg.show()
