import dynesty
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
                  mh=0., fb=0.5, alpha=2.35)


def prior_transform(u):
    """Transforms the uniform random variables `u ~ Unif[0., 1.)`
    to the parameters of interest."""
    x = u.copy()  # copy u
    x_range = [[6.7, 10.0], [15, 22], [0.0, 2.0]]
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
    logage, mh, dm, Av, fb, alpha = 8.4, 0.0, 18.5, 0.4, 0.5, 2.35
    theta_args = logage, dm, Av
    theta = logage, mh, dm, Av, fb, alpha
    step = (0.05, 0.05)  # logage, mh
    n_stars = 1500
    n_syn = 10000
    bins = 50
    h2h_cmd_inst = Hist2Hist4CMD(model, photsys, bins)
    h2p_cmd_inst = Hist2Point4CMD(model, photsys, bins)

    samples = synstars_inst(theta, n_stars, isoc_inst)
    observation = samples  # samples[(samples['g'] < 26.0)]  # samples[(samples['g'] < 25.5) & (samples['i'] < 25.5)]

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
                 mh=mh, fb=fb, alpha=alpha)
    h2p = lnlike(theta_args, step,
                 isoc_inst, h2p_cmd_inst, synstars_inst, n_syn,
                 observation, 'LG', 5,
                 mh=mh, fb=fb, alpha=alpha)
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
    ndim = 3
    likelihood_inst = h2p_cmd_inst

    # sampler = dynesty.NestedSampler(loglike, ptform, ndim)
    # sampler.run_nested()
    # loglike_args = (step, isoc_inst, likelihood_inst, synstars_inst, n_syn, observation, 'LG', 5)
    # Set up the dynamic nested sampling run with multiprocessing
    with dynesty.pool.Pool(20, loglikelihood, prior_transform,
                           logl_args=(step, isoc_inst, likelihood_inst, synstars_inst, n_syn, observation)) as pool:
        sampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform, ndim, pool=pool,
                                        nlive=1000, bound='multi', sample='rwalk')
        sampler.run_nested(dlogz=0.1, checkpoint_file='H2P.3P.save')
        # dlogz=0.01  THE END condition, checkpoint_file='dynesty.save'
