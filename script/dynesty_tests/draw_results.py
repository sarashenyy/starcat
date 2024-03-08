import joblib
import re

import joblib
import numpy as np
from dynesty import plotting as dyplot
from matplotlib import pyplot as plt

# re-defining plotting defaults

plt.style.use('/home/shenyueyue/Projects/starcat/data/mystyle.mplstyle')


# 定义一个函数来提取字符串中的数字
def extract_numbers(text):
    # 使用正则表达式查找所有的数字
    numbers = [float(match) for match in re.findall(r'\d+\.\d+', text)]
    return numbers


logage, mh, dm, Av, fb, alpha = 8.5, 0.0, 10.0, 0.3, 0.3, 2.0
theta = logage, mh, dm, Av, fb, alpha
path_list = [(f'/home/shenyueyue/Projects/starcat/script/dynesty_tests/uncertainty/'
              f'H2P_eps1e-2_0{i + 1}.dsample') for i in range(4)]

label = ['log(age)', '[M/H]', 'DM', '$A_v$', '$f_b$', '$\\alpha$']
ndim = len(label)

test_results = []
for path in path_list:
    dsampler = joblib.load(path)
    results = dsampler.results

    # summary (run) plot
    # fig, axes = dyplot.runplot(results)
    # fig.tight_layout()
    # fig.show()

    # Trace Plots: generate a trace plot showing the evolution of particles
    # (and their marginal posterior distributions) in 1-D projections.
    # colored by importance weight
    # Highlight specific particle paths (shown above) to inspect the behavior of individual particles.
    # (These can be useful to qualitatively identify problematic behavior such as strongly correlated samples.)
    fig, axes = dyplot.traceplot(results, truths=theta,
                                 labels=label,
                                 quantiles=(0.16, 0.5, 0.85),
                                 title_quantiles=(0.16, 0.5, 0.85),
                                 truth_color='black', show_titles=True,
                                 title_kwargs={'fontsize': 24},
                                 trace_cmap='viridis', connect=True,
                                 connect_highlight=range(5)
                                 )  # fig=plt.subplots(6, 2, figsize=(15, 20))
    fig.tight_layout()
    # fig.show()

    res = []
    for i in range(ndim):
        # $\alpha$ = ${1.99}_{-0.07}^{+0.07}$
        aux = axes[:, 1][i].get_title()
        numbers = extract_numbers(aux)
        res.append(numbers)
    # stacked_aux = np.stack(res)
    # print(stacked_aux)
    test_results.append(res)
test_results = np.array(test_results)

xticks = [
    [8.45, 8.50, 8.55, 8.60],  # log(age)
    [-0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2, 0.25],  # [M/H]
    [9.8, 9.9, 10., 10.1],  # DM
    [0.25, 0.3, 0.35, 0.4],  # Av
    [0.25, 0.3, 0.35, 0.4],  # fb
    [1.9, 1.95, 2.0, 2.05, 2.1]  # alpha
]

for i in range(ndim):
    truth = theta[i]
    x = test_results[:, i, 0]
    x_lower = test_results[:, i, 1]
    x_upper = test_results[:, i, 2]
    y = [1, 2, 3, 4]
    colors = ['red', 'green', 'blue', 'orange']
    fig, ax = plt.subplots()

    ax.axvline(x=truth, c='k')
    for j in range(len(x)):
        ax.errorbar(x[j], y[j], xerr=[[x_lower[j]], [x_upper[j]]], fmt='o', capsize=5, color=colors[j])
        ax.text(x[j], y[j] + 0.25, f'{x[j]:.2f}', ha='center')

    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_xticks(xticks[i])
    ax.set_yticklabels([' ', 'test 01', 'test 02', 'test 03', 'test 04', ' '])
    ax.grid(color='gray', linestyle='--')
    ax.set_xlabel(label[i])
    fig.tight_layout()
    fig.show()

# # Corner Points
# # kde=True: colored according to their estimated posterior mass
# # kde=False: colored according to raw importance weights
# fg, ax = dyplot.cornerpoints(results, cmap='plasma', truths=theta,
#                              truth_color='red',
#                              labels=label,
#                              kde=True)
# fg.show()

# Corner Plot
# DEFAULT quantiles=(0.025, 0.5, 0.975)
# fg, ax = dyplot.cornerplot(results, color='blue', truths=theta,
#                            quantiles=(0.16, 0.5, 0.85),
#                            title_quantiles=(0.16, 0.5, 0.85),
#                            labels=label,
#                            truth_color='black', show_titles=True,
#                            max_n_ticks=3)  # quantiles=None
# fg.show()
