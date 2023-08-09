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
from Bio import Align
import multiprocessing
import protein
import get_uniprot
import utils_zy

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

def get_all_ag_hybrid_feature_yakun(coord, chain_mask, atom_mask, cutoff=10):
    atom_num = coord.shape[0]
    p1_coord = coord[~chain_mask]
    p2_coord = coord[chain_mask]

    p1_residue_num = p1_coord.shape[0]
    p2_residue_num = p2_coord.shape[0]

    p1_coord = torch.where(p1_coord!=0, p1_coord, torch.inf * torch.ones_like(p1_coord))
    p2_coord = torch.where(p2_coord!=0, p2_coord, torch.inf * torch.ones_like(p2_coord))
    p1_coord = p1_coord.reshape(-1, 3)
    p2_coord = p2_coord.reshape(-1, 3)

    dists = torch.cdist(p1_coord, p2_coord)

def get_all_ag_hybrid_feature(coord, chain_mask, atom_mask, cutoff=10):
    atom_num = coord.shape[0]
    new_coor = coord.reshape(-1,3)
    
    nonzero_index = torch.nonzero(new_coor==0, as_tuple=False)
    index_tuple = (nonzero_index.t()[0], nonzero_index.t()[1])
    new_coor = new_coor.index_put(index_tuple, torch.ones(nonzero_index.shape[0])*torch.inf)
    dists = torch.cdist(new_coor,new_coor)
    # dists = torch.sum(
    #                         (new_coor[..., None, :] - new_coor[..., None,:, :])
    #                         ** 2,
    #                         -1,
    #                     )
    # dists = torch.sum(
    #                         (coord[...,None,:, None, :] - coord[None,:,None , :, :])
    #                         ** 2,
    #                         -1,
    #                     )
                
    dists = torch.where(torch.isnan(dists), torch.full_like(dists, torch.inf), dists)
    new_dists = dists.reshape(atom_num,37, -1, 37)
    dists = torch.min(torch.min(new_dists, dim=1)[0], dim=-1)[0]
    chain_dist = dists[:, chain_mask][~chain_mask,:]
    return chain_dist


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

# def get_ag_residue_index(pdb_strs, chain_id):
#     index = '-1'
#     residue_index = []
#     n = None
#     for line in pdb_strs:
#         if line[:4] == 'ATOM':
#             if line[21] == chain_id:
#                 if line[23:31].strip() != index:
#                     index = line[23:31].strip()
#                     residue_index.append(int(index))
#     return residue_index

# def get_missing_sequence_3d_atom(ag_original_seq, ag_pdb_seq):

#     missing_3d_seq = ''
#     p_i = 0
#     for o_i in range(len(ag_original_seq)):
#         flag = False
#         if ag_original_seq[o_i] == ag_pdb_seq[p_i]:
#             if p_i < len(ag_pdb_seq) -1:
#                 if ag_original_seq[o_i+1] == ag_pdb_seq[p_i+1] or (p_i != 0 and o_i != 0 and ag_original_seq[o_i-1] == ag_pdb_seq[p_i-1]):
#                     p_i += 1
#                     flag = True
#             elif p_i == len(ag_pdb_seq) -1 and ag_original_seq[o_i-1] == ag_pdb_seq[p_i-1]:
#                 flag = True
#         if flag:
#             missing_3d_seq += ag_original_seq[o_i]
#         else:
#             missing_3d_seq += '-'
#     if len(missing_3d_seq) != len(ag_original_seq) or p_i != len(ag_pdb_seq)-1:
#         print('seq lenght have wrong!!!!!!')
#         return None
#     return missing_3d_seq
        
def alignment_score(str1, str2):
    """
    get similarity score
        :param str1: query sequence
        :param str2: input sequence
        :return: similarity score between 0 and 1
    """
    missing_3d_seq = ''
    aligner = Align.PairwiseAligner()
    alignments = aligner.align(str1, str2)
    alignment = next(alignments)
    for ind, val in enumerate(str1):
        add_flag = True
        for cut_str in alignment.aligned[0]:
            if ind in range(min(cut_str),max(cut_str)):
                missing_3d_seq += val
                add_flag = False
                break
        if add_flag:
            missing_3d_seq += '-'

    return missing_3d_seq

def multi_process_wrapper(worker, proc_num, input_list):
    manager = multiprocessing.Manager()
    return_dict = manager.list()
    jobs = []
    piece = len(input_list) // proc_num + 1
    l = 0
    pdb_loss = pd.read_csv('/nfs_beijing/kubeflow-user/zhangyan_2023/code/result/ab_ag_all_epitope.csv', index_col = None,encoding='gbk')
    pdb_loss_path = pdb_loss['pdb_file_path'].values.tolist()
    # pdb_loss_path = []
    for i in range(proc_num):
        proc_input = input_list[i*piece:(i+1)*piece]
        if len(proc_input) != 0:
            p = multiprocessing.Process(target=worker, args=(return_dict, proc_input, pdb_loss_path))
            l += len(proc_input)
            jobs.append(p)
            p.start()
    for proc in jobs:
        proc.join()
    return return_dict


def process_epitope(np_exsamples, data_items, pdb_loss_path):
    for row in data_items:

        try:
        # if True:
            csv_index_bias = 1 if type(row[0]) == int else 0
            original_seq_by_fasta = utils_zy.get_all_original_seq()
            example = []
            # temp_seqs = row[7].strip().split(',')
            # flag=False
            # for temp_seq in temp_seqs:
            #     if len(set(temp_seq)) ==1:
            #         flag = True
            # if flag == True:
            #     continue
            fname = row[0+csv_index_bias].strip()
            fpath = row[1+csv_index_bias].strip()
            if fpath in pdb_loss_path:
                continue
            chain_ids = row[2+csv_index_bias].strip().split(',')
            # if len(chain_ids) == 1:
            #     continue
            # if 'protein' not in row[7].strip(','):
            #     continue
            # pdb_file_path = os.path.join('/nfs_baoding_ai/sunyiwu/affinity/xtrimo/openfold_debug/metrics_for_all/openmultimer/notebooks/pdb_fv159_ab_ag_cut', fname+'.pdb') # #
            # chain_types = row.chain_type_3d.strip().split(',')
            pdb_file_path = fpath
            print(pdb_file_path)
            with open(pdb_file_path, 'r') as f:
                pdb_strs = f.read()
            ab_len = 0
            all_seq = []
            all_chain_type = []
            all_numbering_seq = []
            all_numbering_chain_type = []
            all_original_seq = []
            all_missing_seq_3d = []
            for i in range(len(chain_ids)):
                #seq = seqs[i]
                chain_id = chain_ids[i]
                _,chain_seq = retrieve_single_chain(pdb_strs.splitlines(), chain_id)
                all_seq.append(chain_seq)
                ab_flag, chain_type = is_antibody(chain_seq)
                all_chain_type.append(chain_type)
                one_original_seq = original_seq_by_fasta['{}_{}'.format(fname, chain_id)]
                all_original_seq.append(one_original_seq)
                all_missing_seq_3d.append(alignment_score(one_original_seq, chain_seq))
                # all_missing_seq_3d.append(get_missing_sequence_3d_atom(one_original_seq, chain_seq))
                if chain_type!='protein':
                    # numbering_seq = make_numbering_by_api(chain_seq)
                    # _, numbering_seq_type = is_antibody(numbering_seq)
                    ...
                else:
                    # numbering_seq = chain_seq
                    # numbering_seq_type = chain_type
                    ag_chain_id = chain_id
                    ag_pdb_seq = chain_seq
                    # ag_residue_index = get_ag_residue_index(pdb_strs.splitlines(), chain_id)
                # all_numbering_seq.append(numbering_seq)
                # all_numbering_chain_type.append(numbering_seq_type)
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
            atom_mask = torch.Tensor(smp_data.atom_mask)
            ag_residue_index = torch.Tensor(smp_data.residue_index)
            ag_residue_index = ag_residue_index[torch.Tensor(smp_data.chain_index) == smp_data.chain_index[-1]]
            
            smp_chain_ids = smp_data.chain_ids

            ag_uniprot_id = get_uniprot.parse_uniprot_id(fname, ag_chain_id)
            ag_uniprot_name = None
            ag_original_seq = all_original_seq[-1]
            original_sequence = ','.join(all_original_seq)
            sequence_3d_atom = ','.join(all_seq)
            missing_sequence_3d_atom = ','.join(all_missing_seq_3d)
            model_sequence = ag_pdb_seq
            model_sequence_renumber = ','.join([str(i) for i in ag_residue_index.long().tolist()])

            example.append(fname)
            example.append(fpath)
            example.append(','.join(chain_ids))
            example.append(ag_uniprot_id)
            example.append(ag_uniprot_name)
            example.append(ag_original_seq)
            example.append(original_sequence)
            example.append(sequence_3d_atom)
            example.append(missing_sequence_3d_atom)
            example.append(model_sequence)
            example.append(model_sequence_renumber)
            
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
                interface_distance = get_all_ag_hybrid_feature(corrd, ag_mask, atom_mask)
                # interface_distance = get_ag_hybrid_feature(corrd, ag_mask)
                # chain_inter = [agid]
                # for cutoff in [10]:
                cutoff = 10
                epitope = torch.sum(interface_distance<=(cutoff), dim=0,keepdims=False).long().squeeze()
                epitope_distance = torch.min(interface_distance, dim=0)[0]
                
                epitope = torch.nonzero((epitope>=1).long()).squeeze()
                pdb_epitope_index = torch.gather(torch.Tensor(ag_residue_index), 0, epitope).long().tolist()
                epitope_and_distance = {}
                epitope_distance = torch.gather(epitope_distance, 0, epitope)
                for dis, epi in zip(epitope_distance, pdb_epitope_index):
                    epitope_and_distance[epi] = dis
                
                epitope_and_distance = sorted(epitope_and_distance.items(), key=lambda x: x[1])
                epitope_distance = ['{}-{:.2f}'.format(idx[0],idx[1]) for idx in epitope_and_distance]
            example.append(','.join(epitope_distance))

        except Exception as e:
            print(e)
            print('error!!!!!!!!!!!!!!!!!!!!!!!!!')
            # break
            continue
        np_exsamples.append(example)
        
        # break


def start():
    # csv_path = '/nfs_beijing/kubeflow-user/zhangyan_2023/code/test_csv/ag_ab_test.csv'
    csv_path = '/nfs_baoding_ai/shaochuan/code/ds_check/ab_ag_training_v6.csv'
    pdb_map = pd.read_csv(csv_path, index_col = None,encoding='gbk')
    print(pdb_map)
    num_worker=30
    smps = np.array(pdb_map)
    print(smps[0])


    add = True
    np_exsamples = multi_process_wrapper(process_epitope, num_worker,smps[:])
    title = ['f_pdb_id', 'pdb_file_path','chain_id','ag_uniprot_id','ag_uniprot_name','ag_original_seq','original_sequence','sequence_3d_atom','missing_sequence_3d_atom','model_sequence','model_sequence_renumber','epitope_distance' ]
    if add:
        df = pd.DataFrame(np.array(np_exsamples), columns = title)
        df.to_csv('result/ab_ag_all_epitope.csv', index=None, mode='a', header=False)
    else:
        df = pd.DataFrame(np.array(np_exsamples), columns = title)
        # print(df)
        df.to_csv('result/ab_ag_all_epitope.csv', index=None)
    #['pdb', 'pdb_file_path','chain','resolution','release_date','seqs','seqs_type', 'fv_seqs','fv_seqs_type', 'stat','interface_min_dis']
    # df.to_csv('code/result/ab_ag_test_all.csv', index=None)


if __name__ == '__main__':
    start()
