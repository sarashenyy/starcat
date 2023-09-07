import matplotlib.pyplot as plt
from .. import config


def draw_isoc(isoc, model, photsyn):
    """
    Draw isochrone and color the stellar period.

    Parameters
    ----------
    isoc : pd.DataFrame
    model : str
    photsyn : str

    Returns
    -------

    """

    color_list = ['grey', 'green', 'orange', 'red', 'blue', 'skyblue', 'pink', 'purple', 'grey', 'black']
    source = config.config[model][photsyn]
    phase = source['phase']

    fig, ax = plt.subplots(figsize=(5, 6))
    for i, element in enumerate(phase):
        if photsyn == 'gaiaDR2' or photsyn == 'gaiaEDR3':
            isoc['color'] = isoc['G_BPmag'] - isoc['G_RPmag']
            ax.plot(isoc[isoc['phase'] == element]['color'], isoc[isoc['phase'] == element]['Gmag'],
                    color=color_list[i], label=element)
            ax.set_xlabel('BP-RP (mag)')
            ax.set_ylabel('G (mag)')
    ax.legend()
    ax.invert_yaxis()
    fig.show()
