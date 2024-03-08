import os

import pandas as pd

folder_path = '/home/shenyueyue/Projects/starcat/data/In100Myr/'
# 获取文件夹中的文件列表
file_list = os.listdir(folder_path)

# 用于存储所有星团信息的字典
data = {}
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    data[file_name] = pd.read_csv(file_path)

# 查看星团基本情况，成员星数
cluster_overview = pd.DataFrame(columns=['file', 'Nstar'])
cluster_overview['file'] = file_list
for i, file_name in enumerate(file_list):
    cluster_overview.iloc[i, 1] = len(data[file_name])
