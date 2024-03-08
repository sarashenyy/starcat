import dynesty
import joblib
from dynesty import plotting as dyplot
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from starcat import (Isoc, Parsec, IMF,
                     BinMRD,
                     CSSTsim, SynStars,
                     lnlike, Hist2Point4CMD, SahaW, EnergyDistance, GaussianKDE)

# re-defining plotting defaults

plt.style.use('/Users/sara/PycharmProjects/starcat/data/mystyle.mplstyle')


def loglikelihood(x, step, isoc_inst, likelihood_inst, synstars_inst, n_syn, observation):
    return lnlike(x, step,
                  isoc_inst, likelihood_inst, synstars_inst, n_syn,
                  observation, 'LG', 10,
                  logage=8., mh=0., dm=18.5, Av=0.5)


def prior_transform(u):
    """Transforms the uniform random variables `u ~ Unif[0., 1.)`
    to the parameters of interest."""
    x = u.copy()  # copy u
    x_range = [[0.0, 1.0], [1.6, 3.0]]
    for i, xrange in enumerate(x_range):
        x[i] = (1 - u[i]) * xrange[0] + u[i] * xrange[1]
    return x


if __name__ == "__main__":
    # initialization
    photsys = 'CSST'
    model = 'parsec'
    imf = 'salpeter55'
    imf_inst = IMF(imf)
    parsec_inst = Parsec()
    binmethod = BinMRD()
    photerr = CSSTsim(model)
    isoc_inst = Isoc(parsec_inst)
    synstars_inst = SynStars(model, photsys,
                             imf_inst, binmethod, photerr)
    logage, mh, dm, Av, fb, alpha = 8., 0.0, 18.5, 0.5, 0.5, 2.35
    theta_args = fb, alpha
    theta = logage, mh, dm, Av, fb, alpha
    step = (0.05, 0.05)  # logage, mh
    logage_step, mh_step = step
    n_stars = 1000
    n_syn = 50000
    # bins = 50
    h2p_cmd_inst = Hist2Point4CMD(model, photsys, bin_method='knuth')
    sahaw_inst = SahaW(model, photsys, bin_method='knuth')
    energy_inst = EnergyDistance(model, photsys, bin_method='knuth')
    kde_inst = GaussianKDE(model, photsys)

    likelihood_inst = h2p_cmd_inst

    samples = synstars_inst(theta, n_stars, isoc_inst)
    observation = samples  # samples[(samples['g'] < 26.0)]  # samples[(samples['g'] < 25.5) & (samples['i'] < 25.5)]
    # observation = samples[samples['i'] < 24]

    obs_isoc = isoc_inst.get_obsisoc(photsys, dm=dm, Av=Av, logage=logage, mh=mh, logage_step=logage_step,
                                     mh_step=mh_step)


    def delta_color(sample, obs_isoc):
        isoc_c = obs_isoc['g'] - obs_isoc['i']
        isoc_m = obs_isoc['i']
        c = sample['g'] - sample['i']
        m = sample['i']
        isoc_line = interp1d(x=isoc_m, y=isoc_c, fill_value='extrapolate')
        temp_c = isoc_line(x=m)
        delta_c = c - temp_c
        sample['delta_c'] = delta_c
        return sample


    # observation = delta_color(observation, obs_isoc)

    # sample_dcmd = synstars_inst.delta_color_samples(theta, n_syn, isoc_inst, logage_step=logage_step, mh_step=mh_step)
    # Dcmd = lnlike(theta_args, step,
    #               isoc_inst, likelihood_inst, synstars_inst, n_syn,
    #               observation, 'LG', 5,
    #               logage=8., mh=0., dm=18.5, Av=0.5)

    fig, ax = plt.subplots(figsize=(4, 5))
    bin = observation['mass_sec'].notna()
    c = observation['g'] - observation['i']
    m = observation['i']

    ax.scatter(c[bin], m[bin], color='#8E8BFE', s=5, alpha=0.6, label='binary')
    ax.scatter(c[~bin], m[~bin], color='#E88482', s=5, alpha=0.6, label='single')
    ax.plot(obs_isoc['g'] - obs_isoc['i'], obs_isoc['i'], color='r')
    ax.set_ylim(max(observation['i']) + 0.2, min(observation['i']) - 0.5)
    ax.set_xlim(min(observation['g'] - observation['i']) - 0.2, max(observation['g'] - observation['i']) + 0.2)
    ax.legend()
    ax.set_title(f'logage={logage}, [M/H]={mh}, DM={dm}, \n'
                 f'Av={Av}, fb={fb}, alpha={alpha}', fontsize=12)
    ax.set_xlabel('g - i')
    ax.set_ylabel('i')

    fig.show()

    # fig, ax = plt.subplots(figsize=(4, 5))
    # bin = sample_dcmd['mass_sec'].notna()
    # dc_syn = sample_dcmd['delta_c']
    # m_syn = sample_dcmd['i']
    # dc_obs = observation['delta_c']
    # m_obs = observation['i']
    #
    # ax.scatter(dc_syn[bin], m_syn[bin], color='#8E8BFE', s=5, alpha=0.6, label='binary')
    # ax.scatter(dc_syn[~bin], m_syn[~bin], color='#E88482', s=5, alpha=0.6, label='single')
    # ax.scatter(dc_obs, m_obs, color='grey', s=5, alpha=0.6)
    # # ax.plot(obs_isoc['g']-obs_isoc['i'], obs_isoc['i'], color='r')
    # # ax.set_ylim(max(observation['i']) + 0.2, min(observation['i']) - 0.5)
    # # ax.set_xlim(min(observation['g'] - observation['i']) - 0.2, max(observation['g'] - observation['i']) + 0.2)
    # ax.invert_yaxis()
    # ax.legend(frameon=True)
    # ax.set_title(f'dCMD_H2P={Dcmd:.2f}', fontsize=12)
    # ax.set_xlabel('g - i')
    # ax.set_ylabel('i')
    #
    # fig.show()

    # Define the dimensionality of our problem.
    ndim = 2

    # sampler = dynesty.NestedSampler(loglike, ptform, ndim)
    # sampler.run_nested()
    # loglike_args = (step, isoc_inst, likelihood_inst, synstars_inst, n_syn, observation, 'LG', 5)
    # Set up the dynamic nested sampling run with multiprocessing
    with dynesty.pool.Pool(20, loglikelihood, prior_transform,
                           logl_args=(step, isoc_inst, likelihood_inst, synstars_inst, n_syn, observation)) as pool:
        # Statistic
        sampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform, ndim, pool=pool,
                                        nlive=1000)  # , bound='single', sample='unif'
        sampler.run_nested(dlogz=0.1, checkpoint_file='H2P.2P.h2p.save')  # checkpoint_file='dynesty.save'
        # Dynamic
        # dsampler = dynesty.DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim, pool=pool,
        #                                         bound='single', sample='unif')  # first_update={'min_eff':25.}
        # dsampler.run_nested(dlogz_init=0.1, nlive_init=500,
        #                     checkpoint_file='H2P.2P.kde100.dsave')
        # nlive_batch=100, maxiter_init=10000, maxiter_batch=1000, maxbatch=10,  n_effective=5000,

    filename = 'H2P.2P.h2p.sampler'
    dir = '/Users/sara/PycharmProjects/starcat/script/dynesty_tests/'
    path = dir + filename
    joblib.dump(sampler, path)

    results = sampler.results

    label = ['$f_b$', '$\\alpha$']
    # summary (run) plot
    fig, axes = dyplot.runplot(results, logplot=True)
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
    #
    # #  Corner Plot
    fg, ax = dyplot.cornerplot(results, color='blue', truths=theta_args,
                               labels=label,
                               truth_color='black', show_titles=True,
                               max_n_ticks=3, quantiles=None)
    fg.show()

    import corner

    fig = corner.corner(results.samples, bins=20, labels=label, truths=[0.5, 2.35], quantiles=[0.16, 0.5, 0.84],
                        show_titles=True)
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # plot 6 snapshots over the course of the run
    for i, a in enumerate(axes.flatten()):
        it = int((i + 1) * results.niter / 8.)
        # overplot the result onto each subplot
        temp = dyplot.boundplot(results, it=it,
                                prior_transform=prior_transform,
                                max_n_ticks=3, show_live=True,
                                dims=(0, 1), labels=['fb', '$\\alpha$'],
                                # span=[(6.7, 10), (-2.0, 0.4)],
                                fig=(fig, a))
        a.set_title('Iteration {0}'.format(it), fontsize=26)
    fig.tight_layout()
    fig.show()
    #
    #
    # # fig, axes = dyplot.cornerbound(results, it=3000,
    # #                                prior_transform=prior_transform,
    # #                                show_live=True)
    # # fig.show()
