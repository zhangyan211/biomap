import os
import gzip
import pandas as pd
import hashlib
from protein import Protein


def get_all_original_seq(fasta_path):
    # fasta_path = '/nfs_baoding/share/shaochuan/seq/pdb_seqres.fasta'
    original_seq = {}
    with open(fasta_path) as fp:
        for line in fp:
            name = line.strip().split()[0][1:]
            name = name.split('_')[0].upper() + '_' + name.split('_')[1]
            seq = fp.readline().strip()
            original_seq[name] = seq
    return original_seq

def get_all_original_seq_new(fasta_path):
    # fasta_path = '/pfs_beijing/share/shaochuan/rcsb_database/v202307/rcsb_protein_sequences/pdb_seqres.txt.gz'
    original_seq = {}
    with gzip.open(fasta_path, 'rt', encoding='utf-8') as fp:
        for line in fp:
            name = line.strip().split()[0][1:]
            name = name.split('_')[0].upper() + '_' + name.split('_')[1]
            message = line[7:].strip()
            seq = fp.readline().strip()
            original_seq[name] = [seq, message]
    return original_seq

def seq_encoder(sequence, method="md5"):
    hasher = eval(f"hashlib.{method}")
    return hasher(sequence.encode(encoding="utf-8")).hexdigest()

def get_ag_seq_original_pdb_path():
    dict_seq = {}
    seq_df = pd.read_csv('/nfs_baoding_ai/shaochuan/code/ds_check/v202307_mab_assembly_v1.csv')
    for index, row in seq_df.iterrows():
        ag_chain = ''
        ag_seq = ''
        for ag_type, chain, seq in zip(row.chain_type.split(','), row.chain_id.split(','), row.seq.split(',')):
            if ag_type == 'protein':
                ag_chain = chain
                ag_seq = seq
        fname = row.pdb_id.split('.')[0] + '_' + ag_chain
        ag_pdb_path = '/pfs_beijing/ai_dataset/xtrimo_dataset/ag_gt_pdb_for_multimer_profiling_hash_new'
        pdb2 = os.path.join(ag_pdb_path, seq_encoder(ag_seq), 'ranked_unrelax_0.pdb')
        dict_seq[fname] = pdb2
    return dict_seq


def get_ag_pdb_path():
    dict_seq = {}
    seq_df = pd.read_csv('/nfs_baoding_ai/shaochuan/code/ds_check/v202307_mab_assembly_v1.csv')
    for index, row in seq_df.iterrows():
        ag_name = row.pdb_id.split('.')[0]
        ag_path = ''
        pdb_id = row.pdb_id
        pdb_num = pdb_id.split('.')[1][3:]
        for ag_type, chain in zip(row.chain_type.split(','), row.chain_id.split(',')):
            if ag_type == 'protein':
                ag_chain = chain
        fname = ag_name + '_' + ag_chain + '_' + pdb_num
        pdb_path = '/pfs_beijing/share/shaochuan/rcsb_database/v202307/ag_single_chain'
        ag_path = ag_name + '_' + pdb_id.split('.')[1][3:] + '_' + ag_chain + '.pdb'
        fpath = os.path.join(pdb_path, ag_path)
        dict_seq[fname] = fpath
    return dict_seq

def get_singal_index(ecd_result_path):
    singal_dict = {}
    with open(ecd_result_path) as fp:
        for line in fp:
            name = line.strip().split()[0][-4:]
            fp.readline()
            pre = fp.readline().strip()
            singal_dict[name] = pre
    return singal_dict

def get_transmembrane_index(ecd_result_path):
    singal_dict = {}
    for i in range(8):
        with open(os.path.join(ecd_result_path,'train_' + str(i),'predicted_topologies.3line')) as fp:
            for line in fp:
                name = line.strip().split()[0][1:]
                fp.readline()
                pre = fp.readline().strip()
                I_flag, O_flag, M_flag = False, False, False
                for ind, value in enumerate(pre):
                    if value == 'I':
                        I_flag = True
                    if value == 'O':
                        O_flag = True
                    if value == 'M':
                        M_flag = True
                if I_flag and O_flag and M_flag:
                    singal_dict[os.path.join('/nfs_baoding/dacheng/B3-process/pdb_v4/pdb_reconstruct_0426', name + '.pdb')] = pre
    return singal_dict


def save_pdb_by_chain(input_pdb_path):
    from Bio.PDB import PDBParser, Select,PDBIO
    p = PDBParser()
    io = PDBIO()
    structure = p.get_structure(None, input_pdb_path)
    res_id = []
    for res in list(structure.get_residues()):
        res_id.append(res.id[1])
    print(res_id)



if __name__ =='__main__':
    # save_pdb_by_chain('/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/test_pdb/6os0_A.pdb', )
    get_ag_pdb_path()