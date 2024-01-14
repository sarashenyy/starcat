import dynesty
from dynesty import plotting as dyplot
from matplotlib import pyplot as plt

from starcat import (Isoc, Parsec, IMF,
                     BinMRD,
                     CSSTsim, SynStars,
                     lnlike, Hist2Hist4CMD, Hist2Point4CMD)

# re-defining plotting defaults

plt.style.use('/Users/sara/PycharmProjects/starcat/data/mystyle.mplstyle')


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
    n_stars = 1500
    n_syn = 100000
    bins = 50
    h2h_cmd_inst = Hist2Hist4CMD(model, photsys, bins)
    h2p_cmd_inst = Hist2Point4CMD(model, photsys, bin_method='knuth')

    samples = synstars_inst(theta, n_stars, isoc_inst)
    observation = samples  # samples[(samples['g'] < 26.0)]  # samples[(samples['g'] < 25.5) & (samples['i'] < 25.5)]
    # observation = samples[samples['i'] < 24]

    fig, ax = plt.subplots(figsize=(4, 5))
    bin = observation['mass_sec'].notna()
    c = observation['g'] - observation['i']
    m = observation['i']

    ax.scatter(c[bin], m[bin], color='blue', s=5, alpha=0.6, label='binary')
    ax.scatter(c[~bin], m[~bin], color='orange', s=5, alpha=0.6, label='single')
    ax.set_ylim(max(observation['i']) + 0.2, min(observation['i']) - 0.5)
    ax.set_xlim(min(observation['g'] - observation['i']) - 0.2, max(observation['g'] - observation['i']) + 0.2)
    ax.legend()
    ax.set_title(f'logage={logage}, [M/H]={mh}, DM={dm}, \n'
                 f'Av={Av}, fb={fb}, alpha={alpha}', fontsize=12)
    ax.set_xlabel('g - i')
    ax.set_ylabel('i')

    fig.show()

    samples = synstars_inst(theta, n_syn, isoc_inst)
    h2h = lnlike(theta_args, step,
                 isoc_inst, h2h_cmd_inst, synstars_inst, n_syn,
                 observation, 'LG', 5,
                 logage=logage, mh=mh, dm=dm, Av=Av)
    h2p = lnlike(theta_args, step,
                 isoc_inst, h2p_cmd_inst, synstars_inst, n_syn,
                 observation, 'LG', 5,
                 logage=logage, mh=mh, dm=dm, Av=Av)
    print(h2h)
    print(h2p)

    fig, ax = plt.subplots(figsize=(4, 4))
    bin = samples['mass_sec'].notna()
    c = samples['g'] - samples['i']
    m = samples['i']
    c_obs = observation['g'] - observation['i']
    m_obs = observation['i']

    ax.scatter(c[bin], m[bin], color='blue', s=5, alpha=0.6)
    ax.scatter(c[~bin], m[~bin], color='orange', s=5, alpha=0.6)
    ax.scatter(c_obs, m_obs, s=1, color='grey')
    ax.invert_yaxis()
    ax.set_xlim(min(c_obs) - 0.2, max(c_obs) + 0.2)
    ax.set_ylim(max(m_obs) + 0.5, min(m_obs) - 0.5)
    ax.set_title(f'logage={logage}, [M/H]={mh}, DM={dm}, \n'
                 f'Av={Av}, fb={fb}, alpha={alpha}', fontsize=12)
    ax.set_xlabel('g - i')
    ax.set_ylabel('i')
    ax.text(0.6, 0.6, f'H2H={h2h:.2f}', transform=ax.transAxes, fontsize=11)
    ax.text(0.6, 0.5, f'H2P={h2p:.2f}', transform=ax.transAxes, fontsize=11)

    fig.show()

    # Define the dimensionality of our problem.
    ndim = 2
    likelihood_inst = h2p_cmd_inst

    # sampler = dynesty.NestedSampler(loglike, ptform, ndim)
    # sampler.run_nested()
    # loglike_args = (step, isoc_inst, likelihood_inst, synstars_inst, n_syn, observation, 'LG', 5)
    # Set up the dynamic nested sampling run with multiprocessing
    with dynesty.pool.Pool(20, loglikelihood, prior_transform,
                           logl_args=(step, isoc_inst, likelihood_inst, synstars_inst, n_syn, observation)) as pool:
        # Statistic
        sampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform, ndim, pool=pool,
                                        nlive=1000, bound='single', sample='unif')
        sampler.run_nested(dlogz=0.01, checkpoint_file='H2P.2Pknuth.save')  # checkpoint_file='dynesty.save'
        # Dynamic
        # dsampler = dynesty.DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim, pool=pool,
        #                                         bound='single', sample='unif')  # first_update={'min_eff':25.}
        # dsampler.run_nested(dlogz_init=0.01, nlive_init=500, n_effective=5000,
        #                     checkpoint_file='H2P.2Pknuth.dsave')
        # nlive_batch=100, maxiter_init=10000, maxiter_batch=1000, maxbatch=10,

    # filename = 'H2P.2Pknuth.dsampler'
    # dir = '/Users/sara/PycharmProjects/starcat/script/dynesty_tests/'
    # path = dir + filename
    # joblib.dump(dsampler, path)
    #
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
    # fig, axes = dyplot.cornerbound(results, it=3000,
    #                                prior_transform=prior_transform,
    #                                show_live=True)
    # fig.show()
