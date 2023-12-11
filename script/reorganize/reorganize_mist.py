import os

import joblib
import pandas as pd

isoc_dir = '/Users/sara/PycharmProjects/starcat/data/isochrones/MIST/HSTWFC3/'
source_dir = '/Users/sara/PycharmProjects/starcat/data/isochrones/MIST/HSTWFC3/MIST_v1.2_vvcrit0.4_HST_WFC3/'

columns = ['EEP', 'log10_isochrone_age_yr', 'initial_mass', 'star_mass', 'log_Teff', 'log_g', 'log_L',
           '[Fe/H]_init', '[Fe/H]', 'WFC3_UVIS_F200LP', 'WFC3_UVIS_F218W', 'WFC3_UVIS_F225W', 'WFC3_UVIS_F275W',
           'WFC3_UVIS_F280N', 'WFC3_UVIS_F300X', 'WFC3_UVIS_F336W', 'WFC3_UVIS_F343N', 'WFC3_UVIS_F350LP',
           'WFC3_UVIS_F373N', 'WFC3_UVIS_F390M', 'WFC3_UVIS_F390W', 'WFC3_UVIS_F395N', 'WFC3_UVIS_F410M',
           'WFC3_UVIS_F438W', 'WFC3_UVIS_F467M', 'WFC3_UVIS_F469N', 'WFC3_UVIS_F475W', 'WFC3_UVIS_F475X',
           'WFC3_UVIS_F487N', 'WFC3_UVIS_F502N', 'WFC3_UVIS_F547M', 'WFC3_UVIS_F555W', 'WFC3_UVIS_F600LP',
           'WFC3_UVIS_F606W', 'WFC3_UVIS_F621M', 'WFC3_UVIS_F625W', 'WFC3_UVIS_F631N', 'WFC3_UVIS_F645N',
           'WFC3_UVIS_F656N', 'WFC3_UVIS_F657N', 'WFC3_UVIS_F658N', 'WFC3_UVIS_F665N', 'WFC3_UVIS_F673N',
           'WFC3_UVIS_F680N', 'WFC3_UVIS_F689M', 'WFC3_UVIS_F763M', 'WFC3_UVIS_F775W', 'WFC3_UVIS_F814W',
           'WFC3_UVIS_F845M', 'WFC3_UVIS_F850LP', 'WFC3_UVIS_F953N', 'WFC3_IR_F098M', 'WFC3_IR_F105W',
           'WFC3_IR_F110W', 'WFC3_IR_F125W', 'WFC3_IR_F126N', 'WFC3_IR_F127M', 'WFC3_IR_F128N', 'WFC3_IR_F130N',
           'WFC3_IR_F132N', 'WFC3_IR_F139M', 'WFC3_IR_F140W', 'WFC3_IR_F153M', 'WFC3_IR_F160W', 'WFC3_IR_F164N',
           'WFC3_IR_F167N', 'phase'
           ]

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
