import numpy as np
import pandas as pd
import os
from shutil import copyfile

excal_path = '/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/data/10416_max_gap_0_30_v2.xlsx'
ag_less_30_data = pd.read_excel(excal_path, sheet_name=1).iloc[:,1]
ag_more_30_data = pd.read_excel(excal_path, sheet_name=2).iloc[:,1]
ag_less_30_name = [name[:6] for name in ag_less_30_data]
ag_more_30_name = [name[:6] for name in ag_more_30_data]
# print(ag_less_30_name)
img_list = os.listdir('/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/less30_img')
img_list = [name[:6] for name in img_list]
for name in img_list:
    if name in ag_less_30_name:
        copyfile(os.path.join('/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/less30_img', name + '.png'), 
                 os.path.join('/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/less30_ag_less', name + '.png'))
    if name in ag_more_30_name:
        copyfile(os.path.join('/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/less30_img', name + '.png'), 
                 os.path.join('/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/less30_ag_more', name + '.png'))
print(len(os.listdir('/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/less30_ag_more')))
print(len(os.listdir('/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/less30_ag_less')))