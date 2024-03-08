import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches

from starcat import (Isoc, Parsec, CSSTsim)

plt.style.use('/Users/sara/PycharmProjects/starcat/data/mystyle.mplstyle')
logage = 10.06
# 读取文件
binary_path = '/Users/sara/PycharmProjects/starcat/data/WangLong/data.2901.binary.chaoliu'
single_path = '/Users/sara/PycharmProjects/starcat/data/WangLong/data.2901.single.chaoliu'
all_path = '/Users/sara/PycharmProjects/starcat/data/WangLong/data.2901.chaoliu'

column_binary = [
    'M1', 'x1', 'y1', 'z1', 'vx1', 'vy1', 'vz1', 'M2', 'x2', 'y2', 'z2', 'vx2', 'vy2', 'vz2',
    'semi-major', 'eccentricity', 'RA', 'Dec', 'Distance', 'pm_RA', 'pm_Dec', 'vr',
    'luminosity1', 'Radius1', 'Temperature1', 'luminosity2', 'Radius2', 'Temperature2',
    'CSST.NUV1', 'CSST.u1', 'CSST.g1', 'CSST.r1', 'CSST.i1', 'CSST.z1', 'CSST.y1',
    'CSST.NUV2', 'CSST.u2', 'CSST.g2', 'CSST.r2', 'CSST.i2', 'CSST.z2', 'CSST.y2'
]
column_single = [
    'Mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'RA', 'Dec', 'Distance', 'pm_RA', 'pm_Dec', 'vr',
    'luminosity', 'Radius', 'Temperature', 'CSST.NUV', 'CSST.u', 'CSST.g', 'CSST.r', 'CSST.i', 'CSST.z', 'CSST.y'
]
column_all = [
    'Mass', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'RA', 'Dec', 'Distance', 'pm_RA', 'pm_Dec', 'vr',
    'luminosity', 'Radius', 'Temperature', 'CSST.NUV', 'CSST.u', 'CSST.g', 'CSST.r', 'CSST.i', 'CSST.z', 'CSST.y'
]

binary_full = pd.read_csv(binary_path, sep=' ', skiprows=1, header=None)
single_full = pd.read_csv(single_path, sep=' ', skiprows=1, header=None)
mix_full = pd.read_csv(all_path, sep=' ', skiprows=1, header=None)
print(binary_full.shape)  # (80951, 42)
print(single_full.shape)  # (47103, 23)
print(mix_full.shape)  # (209000, 23)

mix = mix_full.iloc[:, [0] + list(range(16, len(mix_full.columns)))]
single = single_full.iloc[:, [0] + list(range(16, len(single_full.columns)))]
binary = binary_full.iloc[:, [0] + list(range(28, len(binary_full.columns)))]
columns = ['Mass', 'NUV', 'u', 'g', 'r', 'i', 'z', 'y']
mix.columns = columns
single.columns = columns
binary.columns = ['Mass', 'NUV1', 'u1', 'g1', 'r1', 'i1', 'z1', 'y1', 'NUV2', 'u2', 'g2', 'r2', 'i2', 'z2', 'y2']
for i in range(7):
    binary[columns[i + 1]] = -2.5 * np.log10(
        pow(10, -0.4 * binary.iloc[:, i + 1]) + pow(10, -0.4 * binary.iloc[:, i + 8])
    )

DM = 16.5
conditionb = (((binary['g'] - binary['i']) > -2.7) & ((binary['g'] - binary['i']) < 0.7)
              & (binary['i'] > (5 - DM)) & (binary['i'] < (23 - DM)))
conditions = (((single['g'] - single['i']) > -2.7) & ((single['g'] - single['i']) < 0.7)
              & (single['i'] > (5 - DM)) & (single['i'] < (23 - DM)))

# %%
# add photometric error
model = 'parsec'
photsys = 'CSST'
parsec_inst = Parsec()
isoc_inst = Isoc(parsec_inst)
photerr = CSSTsim(model)

isoc = isoc_inst.get_isoc(photsyn=photsys, logage=logage, mh=-1.4, logage_step=0.05, mh_step=0.05)

# %%
fig, ax = plt.subplots(figsize=(6, 6))
# 定义长方形的左下角坐标 (x, y) 和宽度、高度
x = -2.7
y = 23
width = 3.4
height = -18

# ax.scatter(mix['g'] - mix['i'], mix['i'] + 16.5, s=6)
# ax.scatter(binary['g1'] - binary['i1'], binary['i1'] + 16.5, s=2, color='g')
# ax.scatter(binary['g2'] - binary['i2'], binary['i2'] + 16.5, s=2, color='g')
ax.scatter(binary['g'] - binary['i'], binary['i'] + DM, s=2, color='green', marker='o', label='binary')
ax.scatter(single['g'] - single['i'], single['i'] + DM, s=2, color='orange', marker='o', label='single')
# ax.scatter(binary[~conditionb]['g'] - binary[~conditionb]['i'], binary[~conditionb]['i'] + DM, s=3, color='grey',
#            marker='o')
# ax.scatter(single[~conditions]['g'] - single[~conditions]['i'], single[~conditions]['i'] + DM, s=3, color='grey',
#            marker='o')
ax.plot(isoc['g'] - isoc['i'], isoc['i'] + DM, color='k')
ax.legend(frameon=True)
ax.text(0.55, 0.74, f'log(age) = {logage}', transform=ax.transAxes, c='r')
ax.text(0.55, 0.68, '[M/H] = -1.4', transform=ax.transAxes, c='r')
ax.text(0.55, 0.62, 'DM = 16.5', transform=ax.transAxes, c='r')
ax.text(0.55, 0.56, '$f_b = $' + f'{len(binary[conditionb]) / (len(single[conditions]) + len(binary[conditionb])):.2f}',
        transform=ax.transAxes, c='r')

# 创建一个长方形对象
rectangle = patches.Rectangle((x, y), width, height, edgecolor='r', facecolor='none', linestyle='--', linewidth=1.5)
# 将长方形添加到 Axes 上
# ax.add_patch(rectangle)
# ax.axhline(y=23, color='red', linestyle='--')
# ax.axvline(x=-2.7, color='red', linestyle='--')
ax.invert_yaxis()
ax.set_xlabel('g - i')
ax.set_ylabel('i')
fig.show()
# %%
fig, ax = plt.subplots(figsize=(6, 6))
# 定义长方形的左下角坐标 (x, y) 和宽度、高度
x = -2.7
y = 23
width = 3.4
height = -18

# ax.scatter(mix['g'] - mix['i'], mix['i'] + 16.5, s=6)
# ax.scatter(binary['g1'] - binary['i1'], binary['i1'] + 16.5, s=2, color='g')
# ax.scatter(binary['g2'] - binary['i2'], binary['i2'] + 16.5, s=2, color='g')
ax.scatter(binary['g'] - binary['i'], binary['i'], s=2, color='green', marker='o', label='binary')
ax.scatter(binary[~conditionb]['g'] - binary[~conditionb]['i'], binary[~conditionb]['i'], s=3, color='grey',
           marker='o')
ax.scatter(single['g'] - single['i'], single['i'], s=2, color='orange', marker='o', label='single')
ax.scatter(single[~conditions]['g'] - single[~conditions]['i'], single[~conditions]['i'], s=3, color='grey',
           marker='o')
ax.plot(isoc['g'] - isoc['i'], isoc['i'], color='k')
ax.legend(frameon=True)
ax.text(0.7, 0.74, f'log(age) = {logage}', transform=ax.transAxes, c='r')
ax.text(0.7, 0.68, '[M/H] = -1.4', transform=ax.transAxes, c='r')
ax.text(0.7, 0.62, 'DM = 16.5', transform=ax.transAxes, c='r')
ax.text(0.7, 0.56, '$f_b = $' + f'{len(binary[conditionb]) / (len(single[conditions]) + len(binary[conditionb])):.2f}',
        transform=ax.transAxes, c='r')
# ax.axhline(y=23, color='red', linestyle='--')
# ax.axvline(x=-2.7, color='red', linestyle='--')
ax.invert_yaxis()
ax.set_xlabel('g - i')
ax.set_ylabel('i')
fig.show()
