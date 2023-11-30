import os

import joblib
import pandas as pd

isoc_dir = '/Users/sara/PycharmProjects/starcat/data/isochrones/MIST/UBVRIplus/'
source_dir = '/Users/sara/PycharmProjects/starcat/data/isochrones/MIST/UBVRIplus/MIST_v1.2_vvcrit0.4_UBVRIplus/'
source_file = os.listdir(source_dir)

count = 0
for i, filename in enumerate(source_file):
    # read file
    file_path = source_dir + filename
    # get feh
    feh_part = filename.split('_feh_')[1].split('_')[0]  # 提取feh_m0.25
    feh_value = feh_part.replace('m', '-').replace('p', '+')  # 将m替换为-，p替换为+
    print(feh_value)

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # find start line
    data_start = []
    for j, line in enumerate(lines):
        if line.startswith('# EEP'):
            aux = j
            data_start.append(aux)

    columns = ['EEP', 'log10_isochrone_age_yr', 'initial_mass', 'star_mass', 'log_Teff', 'log_g', 'log_L',
               '[Fe/H]_init', '[Fe/H]', 'Bessell_U', 'Bessell_B', 'Bessell_V', 'Bessell_R', 'Bessell_I',
               '2MASS_J', '2MASS_H', '2MASS_Ks', 'Kepler_Kp', 'Kepler_D51', 'Hipparcos_Hp', 'Tycho_B', 'Tycho_V',
               'Gaia_G_DR2Rev', 'Gaia_BP_DR2Rev', 'Gaia_RP_DR2Rev', 'Gaia_G_MAW', 'Gaia_BP_MAWb', 'Gaia_BP_MAWf',
               'Gaia_RP_MAW', 'TESS', 'Gaia_G_EDR3', 'Gaia_BP_EDR3', 'Gaia_RP_EDR3', 'phase']

    print(f'start to reorganize No.{i} source file, will generate {len(data_start) - 1} files(.joblib) in total.')
    for t in range(len(data_start) - 1):
        start = data_start[t] + 1
        end = data_start[t + 1] - 4  # drop tail rows
        isoc = pd.read_csv(file_path, sep='\s+', skiprows=start, nrows=end - start, header=None)
        isoc.columns = columns

        logage = isoc['log10_isochrone_age_yr'][0]
        isoc_path = isoc_dir + f'age{logage:+.2f}_mh{feh_value}.joblib'
        joblib.dump(isoc, isoc_path)
        count += 1
        if t % 10 == 0:
            print(f'No.{i} source file, {t}/{len(data_start) - 1}')
    print(f'Generated {count} files(.joblib) in total.')
