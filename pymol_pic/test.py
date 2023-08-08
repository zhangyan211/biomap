import pandas as pd
import os 

import json
import numpy as np
import glob
import io
import torch
import gzip
import residue_constants as rc
from Bio.PDB import PDBParser, Select,PDBIO
from Bio import Align
import multiprocessing
import protein



with open('/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/test_pdb/6li3_A.pdb') as fp:
    pdb_str = fp.read()
pdb_strs = ''
for line in pdb_str.split('\n'):
    if line[:4] == 'ATOM':
        pdb_strs += line
        pdb_strs += '\n'


smp_data = protein.from_pdb_string(
                        pdb_strs,
                        chain_id='A',
                        use_filter_atom = True,
                        is_multimer= True,
                        return_id2seq = False,
                        is_cut_ab_fv = True,
                        resolution = 0.0,
                    )
ind = np.argsort(smp_data.residue_index)
# # smp_data.residue_index = smp_data.residue_index[ind]
# with open('/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/test_pdb/test.pdb', 'w') as fpw:
#     fpw.write(protein.to_pdb(smp_data))
# print()

# excal_path = '/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/data/abnb-profile-v17_11723.xlsx'
#     # pdb_map = pd.read_csv(csv_path)
# pdb_map = pd.read_excel(excal_path, sheet_name=2)
# test_list = list(pdb_map.iloc[:,1])

# from collections import Counter   #引入Counter
# a = [i[:-2]for i in test_list]
# b = dict(Counter(a))
# print ([key for key,value in b.items()if value > 1])
print(1 and 1)