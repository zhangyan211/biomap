import pandas as pd
import os 

import json
import pandas as pd
import os
import numpy as np
import os
import pandas as pd
import glob
import io
import torch
import residue_constants as rc
from Bio.PDB import PDBParser, Select,PDBIO
import multiprocessing
import protein
restype_1to3 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
                "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
                "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL","X":"UNK"}
restype_3to1 = {v: k for k, v in restype_1to3.items()}
IMGT = [
    range(1, 27),
    range(27, 39),
    range(39, 56),
    range(56, 66),
    range(66, 105),
    range(105, 118),
    range(118, 129),
]

def regions(seq, numbering, rng):
    fr1, fr2, fr3, fr4 = [], [], [], []
    cdr1, cdr2, cdr3 = [], [], []
    type_list = []

    for item in numbering[0][0][0]:
        (idx, key), aa = item
        sidx = "%d%s" % (idx, key.strip())  # str index
        if idx in rng[0]:  # fr1
            fr1.append([sidx, aa])
            type_list.append("fr1")
        elif idx in rng[1]:  # cdr1
            cdr1.append([sidx, aa])
            type_list.append("cdr1")
        elif idx in rng[2]:  # fr2
            fr2.append([sidx, aa])
            type_list.append("fr2")
        elif idx in rng[3]:  # cdr2
            cdr2.append([sidx, aa])
            type_list.append("cdr2")
        elif idx in rng[4]:  # fr3
            fr3.append([sidx, aa])
            type_list.append("fr3")
        elif idx in rng[5]:  # cdr3
            type_list.append("cdr3")
            cdr3.append([sidx, aa])
        elif idx in rng[6]:  # fr4
            fr4.append([sidx, aa])
            type_list.append("fr4")
        else:
            pass
            # logger.info(f"[WARNING] seq={seq}, sidx={sidx}, aa={aa}")

    return fr1, fr2, fr3, fr4, cdr1, cdr2, cdr3, type_list
from anarci import anarci
def make_numbering_by_api(
    seq, input_species=None, scheme="imgt", input_chain_type=None, ncpu=4
):
    seqs = [("0", seq)]
    numbering, alignment_details, hit_tables = anarci(
        seqs, scheme=scheme, output=False, ncpu=ncpu
    )
  

    species = alignment_details[0][0]["species"].lower()
    chain_type = alignment_details[0][0]["chain_type"].lower()
    e_value = alignment_details[0][0]["evalue"]
    score = alignment_details[0][0]["bitscore"]
    v_start = alignment_details[0][0]["query_start"]
    v_end = alignment_details[0][0]["query_end"]
    if scheme == "imgt":
        rng = IMGT
    elif scheme == "kabat" and chain_type.lower() == "h":
        rng = KABAT_H
    elif scheme == "kabat" and (chain_type.lower() == "l" or chain_type.lower() == "k"):
        rng = KABAT_L
    else:
        raise NotImplementedError
    fr1, fr2, fr3, fr4, cdr1, cdr2, cdr3, type_list = regions(seq, numbering, rng)
    str_fr1 = "".join([item[1] for item in fr1 if item[1] != "-"])
    str_fr2 = "".join([item[1] for item in fr2 if item[1] != "-"])
    str_fr3 = "".join([item[1] for item in fr3 if item[1] != "-"])
    str_fr4 = "".join([item[1] for item in fr4 if item[1] != "-"])
    str_cdr1 = "".join([item[1] for item in cdr1 if item[1] != "-"])
    str_cdr2 = "".join([item[1] for item in cdr2 if item[1] != "-"])
    str_cdr3 = "".join([item[1] for item in cdr3 if item[1] != "-"])
    str_overall = str_fr1 + str_cdr1 + str_fr2 + str_cdr2 + str_fr3 + str_cdr3 + str_fr4
    return str_overall

def is_antibody(seq, scheme='imgt', ncpu=4):
    seqs = [("0", seq)]
    numbering, alignment_details, hit_tables = anarci(seqs, scheme=scheme, output=False, ncpu=ncpu)
    if numbering[0] is None:
        return False, 'protein'
    chain_type = alignment_details[0][0]['chain_type'].lower()

    if chain_type is None:
        return False, 'protein'
    else:
        return True, chain_type


def get_ag_hybrid_feature(coord, chain_mask, cutoff=10):
    CA_index = rc.atom_order["CA"]
    # print(coord.shape   )
    CA_coor = coord[...,CA_index,:]
    # print(CA_coor.shape   )
    # dists = 1e-16 + torch.sum((CA_coor[..., None, :] - CA_coor[..., None, :, :])
    #                     ** 2,-1) 
    dists = torch.sqrt(
                    1e-16
                    + torch.sum(
                        (CA_coor[..., None, :] - CA_coor[..., None, :, :])
                        ** 2,
                        -1,
                    )
                ) 
    dists = torch.sum(
                            (CA_coor[..., None, :] - CA_coor[..., None, :, :])
                            ** 2,
                            -1,
                        )
                
    chain_dist = dists[:, chain_mask][~chain_mask,:]
    # chain_dist = chain_dist**0.5
    # print(chain_dist)
    # print(chain_dist.shape)
    # print(chain_dist.min(dim=0))
    # print(chain_dist.argmin(dim=0))
    # print(chain_dist.min())
    return chain_dist
    # epitope = torch.sum(chain_dist<=(cutoff**2), dim=0,keepdims=False).long()
    # epitope = (epitope>1).long()
    # epitope = epitope*8
    # epitope_feature = F.one_hot(
    #     epitope, num_classes=9
    # )
    # chain_type_feature = torch.zeros((epitope_feature.shape[0], 4))
    # chain_type_feature[0:]=torch.Tensor([0, 0, 0, 1])
    # aatype = make_one_hot(chain_aatype, 21)
    # concat_feature = torch.cat([epitope_feature, chain_type_feature, aatype], dim=-1).to(torch.float32)
    # return concat_feature


def retrieve_single_chain(pdb_strs, chain_id):
    atoms = []
    pdb_lines = []
    n = None
    for line in pdb_strs:
        if line[:4] == 'ATOM':
            if line[21] == chain_id:
                pdb_lines.append(line)
                if line[22:27] != n:
                    if line[17:20] in restype_3to1:
                        atoms.append(restype_3to1[line[17:20]])
                    else:
                        atoms.append('X')
                    n = line[22:27]
    return pdb_lines, ''.join(atoms)

def multi_process_wrapper(worker, proc_num, input_list):
    manager = multiprocessing.Manager()
    return_dict = manager.list()
    jobs = []
    piece = len(input_list) // proc_num + 1
    l = 0
    for i in range(proc_num):
        proc_input = input_list[i*piece:(i+1)*piece]
        if len(proc_input) != 0:
            p = multiprocessing.Process(target=worker, args=(return_dict, proc_input))
            l += len(proc_input)
            jobs.append(p)
            p.start()
    for proc in jobs:
        proc.join()
    return return_dict

csv_path = '/nfs_baoding_ai/sunyiwu/affinity/xtrimo/openfold_debug/metrics_for_all/openmultimer/notebooks/fv_159.csv'
#'/nfs_baoding_ai/sunyiwu/affinity/xtrimo/openfold_debug/metrics_for_all/openmultimer/notebooks/fv_159.csv'
pdb_map = pd.read_csv(csv_path, index_col = None,encoding='gbk')
print(pdb_map)

def process_epitope(np_exsamples, data_items):
    for row in data_items:

        try:
        # if True:
            example = []
            temp_seqs = row[6].strip().split(',')
            flag=False
            for temp_seq in temp_seqs:
                if len(set(temp_seq)) ==1:
                    flag = True
            if flag == True:
                continue
            fname = row[0].strip()
            example.append(fname)

            chain_ids = row[2].strip().split(',')
            if len(chain_ids) == 1:
                continue
            if 'protein' not in row[3].strip():
               
                continue
            pdb_file_path = os.path.join('/nfs_baoding_ai/sunyiwu/affinity/xtrimo/openfold_debug/metrics_for_all/openmultimer/notebooks/pdb_fv159_ab_ag_cut', fname+'.pdb') # #
            # chain_types = row.chain_type_3d.strip().split(',')
            print(pdb_file_path)
            with open(pdb_file_path, 'r') as f:
                pdb_strs = f.read()
            all_seq = []
            all_chain_type = []
            all_numbering_seq = []
            all_numbering_chain_type = []
            for i in range(len(chain_ids)):
                #seq = seqs[i]
                chain_id = chain_ids[i]
                _,chain_seq = retrieve_single_chain(pdb_strs.splitlines(), chain_id)
                all_seq.append(chain_seq)
                ab_flag, chain_type = is_antibody(chain_seq)
                all_chain_type.append(chain_type)
                if chain_type!='protein':
                    numbering_seq = make_numbering_by_api(chain_seq)
                    _, numbering_seq_type = is_antibody(numbering_seq)
                else:
                    numbering_seq = chain_seq
                    numbering_seq_type = chain_type
                all_numbering_seq.append(numbering_seq)
                all_numbering_chain_type.append(numbering_seq_type)
            smp_data = protein.from_pdb_string(
                        pdb_strs,
                        chain_id=chain_ids,
                        use_filter_atom = True,
                        is_multimer= True,
                        return_id2seq = False,
                        is_cut_ab_fv = True,
                        resolution = 0.0,
                    )
            corrd = torch.Tensor(smp_data.atom_positions)
            
            smp_chain_ids = smp_data.chain_ids
            example.append(chain_ids)
            example.append(all_chain_type)
            example.append(all_seq)
            all_chain_epitope = []
            for i in range(len(chain_ids)):
                if all_chain_type[i] == 'protein':
                    agid = chain_ids[i]
                else:
                    continue
                ag_mask =[]
                for idx in smp_chain_ids:
                    if agid==idx:
                        ag_mask.append(1)
                    else:
                        ag_mask.append(0)
                ag_mask = torch.Tensor(ag_mask)
                ag_mask = ag_mask==1
                interface_distance = get_ag_hybrid_feature(corrd, ag_mask)
                chain_inter = [agid]
                for cutoff in [8,10,12,14]:
                    epitope = torch.sum(interface_distance<=(cutoff**2), dim=0,keepdims=False).long()
                    #epitope = (epitope>1)
                    
                    epitope = torch.nonzero((epitope>1).long()).squeeze().tolist()
                    # epitope = (epitope>1).long().squeeze().tolist()
                    if isinstance(epitope,int):
                        epitope=[epitope]
                    epitope = [str(idx) for idx in epitope]
                    chain_inter.append(','.join(epitope))

                    # print(epitope)
                    # print(epitope.shape)

                    # exit()
                all_chain_epitope.append('-'.join(chain_inter))
            example.append(':'.join(all_chain_epitope))

        except Exception as e:
            print(e)
            continue
        
        # example.append(row.pdb)
        # example.append(pdb_file_path)
        # example.append(row.chain_id.strip())
        # example.append(row.r_precision)
        # example.append(row.release_date)
        # example.append(','.join(all_seq))
        # example.append(','.join(all_chain_type))
        # example.append(','.join(all_numbering_seq))
        # example.append(','.join(all_numbering_chain_type))

        # state = 1
        # if ','.join(all_chain_type) != ','.join(all_numbering_chain_type):
        #     state = 0
        # if ','.join(all_chain_type) not in ['h,k,protein','h,l,protein']:
        #     state=2
        # example.append(state)
        # example.append(interface_min_dis.item())
        np_exsamples.append(example)
        # if len(np_exsamples)>10:
        #     break
num_worker=30
smps = np.array(pdb_map)
print(smps[0])
np_exsamples = multi_process_wrapper(process_epitope, num_worker,smps[:])
title = ['pdb','chain_ids','chain_types','seq','epitope' ]#['pdb', 'pdb_file_path','chain','resolution','release_date','seqs','seqs_type', 'fv_seqs','fv_seqs_type', 'stat','interface_min_dis']

df = pd.DataFrame(np.array(np_exsamples), columns = title)
print(df)
df.to_csv('fv_159_v3_epitope.csv')