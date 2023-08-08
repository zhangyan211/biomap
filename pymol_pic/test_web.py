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
# import tool_data
import hashlib


text_yaml = []
text_yaml.append('setTransparency [A 9-1316] [1]')
text_yaml.append('setColor [A 9-323] [0x0000ff]')
text_yaml.append('setColor [A 1225-1316] [0x00ff00]')
display_config_path = '/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/test_yaml/6os0_A.yaml'
with open(display_config_path, 'w') as df:
    df.write('\n'.join(text_yaml))
    
def seq_encoder(sequence, method="md5"):
    hasher = eval(f"hashlib.{method}")
    return hasher(sequence.encode(encoding="utf-8")).hexdigest()

ag_seq = 'WKEAKTTLFCASDAKAYEKECHNVWATHACVPTDPNPQEVVLEQVTENFNMWKNDMVDQMQEDVISIWDQCLKPCVKLTNTSTLTQACPKVTFDPIPIHYCAPAGYAILKCNNKTFNGKGPCNNVSTVQCTHGIKPVVSTQLLLNGSLAEEEIVIRSKNLRDNAKIIIVQLQKSVEIVCTRPNNGGSGSGGDIRQAYCQISGRNWSEAVNQVKKKLKEHFPHKNISFQSSSGGDLEITTHSFNCGGEFFYCNTSGLFQDTISNATIMLPCRIKQIINMWQEVGKAIYAPPIKGQITCKSDITGLLLLRDGGDTTDNTEIFRPSGGDMRDNWRSELYKYKV'
ag_pdb_path = '/pfs_beijing/ai_dataset/xtrimo_dataset/ag_gt_pdb_for_multimer_profiling_hash_new'
pdb2 = os.path.join(ag_pdb_path, seq_encoder(ag_seq), 'ranked_unrelax_0.pdb')
print(pdb2)
# pdb1 = '/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/test_pdb/8hcx_C.pdb'
# http_path2 = 'https://os.biomap-int.com/web/tools/#/structure/protein-structure'
# http_path = http_path2 + f'?file={pdb2}' + f'&display_config={display_config_path}'
# print(http_path)