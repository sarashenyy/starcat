import joblib
import numpy as np

# logage = [7., 8., 9.]
# mh = [-0.5, 0]
# dm = [18.48, 18.94, 24.47, 24.54]
# Av = [0.02, 0.5, 1.]
# fb = [0.2, 0.4, 0.6]
# alpha = [1.8, 2.35, 2.8]
logage = np.array([7., 8., 9.])
mh = np.array([0])
dm = np.array([22])  # [18.5, 20]
Av = np.array([0.05])
fb = np.array([0.35])
alpha = np.array([1.8, 2.35, 2.8])
times = 1

aa, bb, cc, dd, ee, ff, gg = np.indices((len(logage), len(mh), len(dm), len(Av), len(fb), len(alpha), times))
aa = aa.ravel()
bb = bb.ravel()
cc = cc.ravel()
dd = dd.ravel()
ee = ee.ravel()
ff = ff.ravel()
gg = gg.ravel()

logage_vals = logage[aa]
mh_vals = mh[bb]
dm_vals = dm[cc]
Av_vals = Av[dd]
fb_vals = fb[ee]
alpha_vals = alpha[ff]

param = np.column_stack((logage_vals, mh_vals, dm_vals, Av_vals, fb_vals, alpha_vals))

# 将数据均分为cut_num个部分
cut_num = 3
split_param = np.array_split(param, cut_num, axis=0)
# 保存每个部分的数据
for i, part in enumerate(split_param):
    joblib.dump(part, f'/Users/sara/PycharmProjects/starcat/script/validation/param{i}.joblib')
