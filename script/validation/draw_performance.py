import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

plt.style.use('/Users/sara/PycharmProjects/starcat/data/mystyle.mplstyle')

data0 = joblib.load('/Users/sara/PycharmProjects/starcat/script/validation/data0.joblib')
data1 = joblib.load('/Users/sara/PycharmProjects/starcat/script/validation/data1.joblib')
data2 = joblib.load('/Users/sara/PycharmProjects/starcat/script/validation/data2.joblib')

data = np.vstack((data0, data1, data2))
print(data.shape)  # (number of cluster samples, [logage, mh, dm, Av, fb, alpha], [true, 0.16, 0.5, 0.84])

age = data[:, 0, 2]
age_true = data[:, 0, 0]
age_err = np.vstack((data[:, 0, 2] - data[:, 0, 1], data[:, 0, 3] - data[:, 0, 2]))
dm = data[:, 2, 2]
dm_true = data[:, 2, 0]
dm_err = np.vstack((data[:, 2, 2] - data[:, 2, 1], data[:, 2, 3] - data[:, 2, 2]))
alpha = data[:, 5, 2]
alpha_true = data[:, 5, 0]
alpha_err = np.vstack((data[:, 5, 2] - data[:, 5, 1], data[:, 5, 3] - data[:, 5, 2]))

# draw performance: {dm, age}, {alpha, age}, {alpha, dm}
# fig data
x, y = alpha, dm
x_err, y_err = alpha_err, dm_err
x_true, y_true = alpha_true, dm_true
xlabel, ylabel = '$\\alpha$', 'DM'

# fig setting
fig = plt.figure(1, figsize=(7, 6))
gs = GridSpec(4, 4)
gs.update(hspace=0.05, wspace=0.05)

# main
ax_main = plt.subplot(gs[1:, :-1])
ax_main.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', capsize=5)
for xi in set(x_true):
    ax_main.axvline(x=xi, color='grey', alpha=0.5, linestyle='-')
for yi in set(y_true):
    ax_main.axhline(y=yi, color='grey', alpha=0.5, linestyle='-')
ax_main.set_xlabel(xlabel)
ax_main.set_ylabel(ylabel)

# upper
ax_upper = plt.subplot(gs[0, :-1], sharex=ax_main)
ax_upper.axhline(y=0, color='grey', alpha=0.5, linestyle='-')

unique_x = np.unique(x_true)
mean_diff_list = []
for val in unique_x:
    indices = np.where(x_true == val)  # 找到x_true中等于当前唯一值的索引
    diff = x[indices] - val  # 计算x - x_true
    mean_diff = np.mean(diff)  # 求平均
    mean_diff_list.append(mean_diff)
    ax_upper.axvline(x=val, color='grey', alpha=0.5, linestyle='-')

ax_upper.scatter(unique_x, mean_diff_list)
ax_upper.set_ylim(-(max(np.abs(mean_diff_list)) * 1.5), +(max(np.abs(mean_diff_list)) * 1.5))
ax_upper.xaxis.set_visible(False)
ax_upper.set_ylabel('$\Delta$' + xlabel)

# right
ax_right = plt.subplot(gs[1:, -1], sharey=ax_main)
ax_right.axvline(x=0, color='grey', alpha=0.5, linestyle='-')

unique_y = np.unique(y_true)
mean_diff_list = []
for val in unique_y:
    indices = np.where(y_true == val)  # 找到x_true中等于当前唯一值的索引
    diff = y[indices] - val  # 计算x - x_true
    mean_diff = np.mean(diff)  # 求平均
    mean_diff_list.append(mean_diff)
    ax_right.axhline(y=val, color='grey', alpha=0.5, linestyle='-')

ax_right.scatter(mean_diff_list, unique_y)
ax_right.set_xlim(-(max(np.abs(mean_diff_list)) * 1.5), +(max(np.abs(mean_diff_list)) * 1.5))
ax_right.yaxis.set_visible(False)
ax_right.set_xlabel('$\Delta$' + ylabel)

fig.show()
