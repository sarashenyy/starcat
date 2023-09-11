import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import Parallel, delayed

from ..likelihood import lnlike_2p


def lnlike_2p_distrib(fb, dm, isoc, likelihoodfunc, synstars, logage_grid, mh_grid, sample_obs, n_jobs=20):
    astart, aend, astep = logage_grid
    mstart, mend, mstep = mh_grid
    step = (astep, mstep)
    abin = np.arange(astart, aend, astep)
    mbin = np.arange(mstart, mend, mstep)
    logage_mh = []
    for a in abin:
        for m in mbin:
            logage_mh.append([a, m])
    print(f"calculate in total : {len(logage_mh)} lnlike values")

    # nested function, access variable in parent function
    def lnlike_wrapper(theta_part):
        lnlikelihood = lnlike_2p(theta_part, fb, dm, step, isoc, likelihoodfunc, synstars, sample_obs)
        return lnlikelihood

    # parallel excution
    results = Parallel(n_jobs=n_jobs)(
        delayed(lnlike_wrapper)(theta_2p) for theta_2p in logage_mh
    )

    # Create DataFrame
    df = pd.DataFrame(logage_mh, columns=['logage', 'mh'])
    df['lnlike'] = results
    func = likelihoodfunc.func
    binm = synstars.binmethod.method
    df.to_csv(f'/home/shenyueyue/Projects/starcat/test_data/{func}_{binm}.csv', index=False)
    return df


def lnlike_distrub_3d(data, type):
    logage = data['logage']
    mh = data['mh']
    lnlike = data['lnlike']

    # 创建散点图
    scatter = go.Scatter3d(
        x=logage,
        y=mh,
        z=lnlike,
        mode='markers',
        marker=dict(
            size=3,
            color=lnlike,
            colorscale='Viridis',
            showscale=True
        )
    )

    # 找到lnlike最大值的索引
    max_index = np.argmax(lnlike)
    max_logage = logage[max_index]
    max_mh = mh[max_index]
    max_lnlike = lnlike[max_index]

    # 添加最大lnlike的点
    scatter_max = go.Scatter3d(
        x=[max_logage],
        y=[max_mh],
        z=[max_lnlike],
        mode='markers',
        marker=dict(
            size=5,
            color='red'
        ),
        name='Max lnlike'
    )

    # 创建图形布局
    layout = go.Layout(
        title=f'{type} Lnlike Distribution',
        scene=dict(
            xaxis_title='logage',
            yaxis_title='mh',
            zaxis_title='lnlike'
        ),
        width=800,  # 设置图形宽度
        height=600  # 设置图形高度
    )

    # 创建图形对象
    fig = go.Figure(data=[scatter, scatter_max], layout=layout)

    # 显示图形
    fig.show()
