from pathlib import Path

import pandas as pd
import numpy as np
import torch



def test1():
    pdb_loss = pd.read_csv('/nfs_beijing/kubeflow-user/zhangyan_2023/code/result/ab_ag_all_epitope.csv', index_col = None,encoding='gbk')
    pdb_loss_path = pdb_loss['pdb_file_path'].values.tolist()
    print(len(set(pdb_loss_path)))


if __name__ == '__main__':
    test1()