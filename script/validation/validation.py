import sys
import time
from multiprocessing import Pool

import corner
import emcee
import joblib
import numpy as np

from starcat import (Isoc, Parsec, IMF,
                     BinMRD,
                     CSSTsim, SynStars,
                     lnlike, Hist2Point4CMD)

# plt.style.use('/Users/sara/PycharmProjects/starcat/data/mystyle.mplstyle')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python validation.py <input_file> <output_file>')
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    param = joblib.load(input_file_path)

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
    # lnlike
    bins = 50
    h2p_cmd_inst = Hist2Point4CMD(model, photsys, bins)

    n_stars = 1000
    n_syn = 10000
    # logage, mh step
    step = (0.05, 0.05)
    # mcmc
    ndim = 6
    nwalkers = 60

    data_list = []

    print(f'total: {len(param)} mock clusters')
    start = time.time()
    for j in range(len(param)):
        # make mock cluster for each theta
        theta = tuple(param[j])
        logage, mh, dm, Av, fb, alpha = theta
        samples = synstars_inst(theta, n_stars, isoc_inst)
        observation = samples
        joblib.dump(observation,
                    f'/Users/sara/PycharmProjects/starcat/script/validation/cat/age{logage:.2f}_mh{mh:.2f}_dm{dm:.2f}_Av{Av:.2f}_fb{fb:.2f}_alp{alpha:.2f}.joblib')
        print(f'the {j}th mock cluster: age{logage:.2f}_mh{mh:.2f}_dm{dm:.2f}_Av{Av:.2f}_fb{fb:.2f}_alp{alpha:.2f}')

        # make initial p0 for mcmc
        temp = []
        scale = np.array([0.4, 0.1, 0.8, 0.5, 0.08, 0.2])
        theta_range = [[6.7, 10.0], [-1.0, 0.4],
                       [17.5, 25.5], [0.0, 2.0],
                       [0.2, 0.9], [1.6, 3.0]]
        for i in range(ndim):
            aux_list = []
            while len(aux_list) < nwalkers:
                aux = theta[i] + scale[i] * np.random.randn()
                if theta_range[i][0] <= aux <= theta_range[i][1]:
                    aux_list.append(aux)
            temp.append(aux_list)
        p0 = np.array(temp).T

        print(f'the {j}th mock cluster: start mcmc')
        likelihood_inst = h2p_cmd_inst
        with Pool(10) as pool:  # local:10(20min) ; raku:30(20min)
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, lnlike,
                pool=pool,
                args=(step, isoc_inst, likelihood_inst, synstars_inst, n_stars, observation, 'LG', 5)
            )
            nburn = 1000
            pos, prob, state = sampler.run_mcmc(p0, nburn, progress=True)
        print(f'the {j}th mock cluster: end mcmc, calculate result')
        # calculate result
        result = None
        for i in range(ndim):
            temp = corner.quantile(sampler.flatchain[:, i], [0.16, 0.5, 0.84])
            if result is None:
                result = temp
            else:
                result = np.vstack((result, temp))

        # combine theta and mcmc result
        data = np.column_stack((theta, result))
        data_list.append(data)
        # save data
        joblib.dump(data_list, output_file_path)
        print(f'result saved to {output_file_path}')

        # save mcmc samples
        joblib.dump(sampler.flatchain,
                    f'/Users/sara/PycharmProjects/starcat/script/validation/mcmc/age{logage:.2f}_mh{mh:.2f}_dm{dm:.2f}_Av{Av:.2f}_fb{fb:.2f}_alp{alpha:.2f}.joblib')
        print('mcmc samplers saved')

    end = time.time()
    execution_time = end - start
    print(f'total cluster: {len(param)} mock clusters done. \n'
          f'total time: {execution_time} s')
