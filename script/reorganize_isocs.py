# this is a script for reorganize the .dat file download from http://stev.oapd.inaf.it/cgi-bin/cmd
import os

import joblib
import pandas as pd

data_dir = '/Users/sara/PycharmProjects/starcat/data/'
isoc_dir = 'isochrones/parsec/CSST/'

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

    columns = [
        'Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe', 'logg', 'label',
        'McoreTP', 'C_O', 'period0', 'period1', 'period2', 'period3', 'period4', 'pmode',
        'Mloss', 'tau1m', 'X', 'Y', 'Xc', 'Xn', 'Xo', 'Cexcess', 'Z', 'mbolmag', 'NUVmag',
        'umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'
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
