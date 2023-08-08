import pandas as pd
import os 

import json
import numpy as np
import glob
import io
# import torch
import gzip
import residue_constants as rc
from Bio.PDB import PDBParser, Select,PDBIO
from Bio import Align
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm
# from pymol import cmd
import protein
import hashlib
import utils_zy
import utils

pdb_path_dict = utils_zy.get_ag_seq_original_pdb_path()
seq_df = pd.read_csv('/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/data/data_profile_v10.csv')
examples = []
for index, row in seq_df.iterrows():
    example = []
    example.append(row.id)
    example.append(row.seg)
    example.append(row.start)
    example.append(row.end)
    example.append(row.gap)
    example.append(row.max_gap)
    example.append(row.status)
    example.append(row.chain_type)
    path = pdb_path_dict[row.id]
    if os.path.exists(path):
        example.append(path)
    else:
        example.append('')
    examples.append(example)

title = ['id', 'seg', 'start','end', 'gap', 'max_gap', 'status', 'chain_type', 'original_pdb_path']
df = pd.DataFrame(np.array(examples), columns = title)
print(df)
df.to_csv('/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/data/data_profile_zy.csv', index=None)
