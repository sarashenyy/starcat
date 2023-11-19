# this is a script for reorganize the .dat file download from http://stev.oapd.inaf.it/cgi-bin/cmd
import os

import joblib
import pandas as pd

photsys = 'gaia'  # 'CSST'
data_dir = '/Users/sara/PycharmProjects/starcat/data/'
isoc_dir = 'isochrones/parsec/gaiaDR3/'

# download source .dat file from http://stev.oapd.inaf.it/cgi-bin/cmd by hand
source_file = os.listdir(data_dir + isoc_dir + 'source/')

count = 0
for i, filename in enumerate(source_file):
    # read file
    file_path = data_dir + isoc_dir + 'source/' + filename
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # find start line
    data_start = []
    for j, line in enumerate(lines):
        if line.startswith('# Zini'):
            aux = j
            data_start.append(aux)
    if photsys == 'CSST':
        columns = [
            'Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe', 'logg', 'label',
            'McoreTP', 'C_O', 'period0', 'period1', 'period2', 'period3', 'period4', 'pmode',
            'Mloss', 'tau1m', 'X', 'Y', 'Xc', 'Xn', 'Xo', 'Cexcess', 'Z', 'mbolmag', 'NUVmag',
            'umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'
        ]
    elif photsys == 'gaia':
        columns = [
            'Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe', 'logg', 'label', 'McoreTP', 'C_O',
            'period0', 'period1', 'period2', 'period3', 'period4', 'pmode', 'Mloss', 'tau1m', 'X', 'Y', 'Xc', 'Xn',
            'Xo', 'Cexcess', 'Z', 'mbolmag', 'Gmag', 'G_BPmag', 'G_RPmag'
        ]
    # save data to pd.df, drop the last line of each file
    print(f'start to reorganize No.{i} source file, will generate {len(data_start) - 1} files(.joblib) in total.')
    for t in range(len(data_start) - 1):
        start = data_start[t] + 1
        end = data_start[t + 1] - 1  # drop the last point (may be wrong point)
        isoc = pd.read_csv(file_path, sep='\s+', skiprows=start, nrows=end - start, header=None)
        isoc.columns = columns

        logage = isoc['logAge'][0]
        mh = isoc['MH'][0]
        isoc_path = data_dir + isoc_dir + f'age{logage:+.2f}_mh{mh:+.2f}.joblib'
        joblib.dump(isoc, isoc_path)
        count += 1
        if t % 10 == 0:
            print(f'No.{i} source file, {t}/{len(data_start) - 1}')
print(f'Generated {count} files(.joblib) in total.')

# if MH is -0.0 in files
# import os
# folder_path = '/Users/sara/PycharmProjects/starcat/data/isochrones/parsec/gaiaDR3/'  # 替换为你的文件夹路径
# for filename in os.listdir(folder_path):
#     if '-0.00' in filename:
#         new_filename = filename.replace('-0.00', '+0.00')
#         old_filepath = os.path.join(folder_path, filename)
#         new_filepath = os.path.join(folder_path, new_filename)
#         os.rename(old_filepath, new_filepath)
