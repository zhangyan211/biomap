import pandas as pd
import os 
import PIL.Image as Image
import shutil

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
from pymol import cmd
import protein
import hashlib
import utils_zy
import utils



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

def ag_retrieve_single_chain(pdb_strs, chain_id):
    pdb_strs = pdb_strs.split('\n')
    pdb_lines = []
    atoms = []
    index_resident = []
    n = None
    start_ind = -1
    for line in pdb_strs:
        if line[:4] == 'ATOM':
            if line[21] == chain_id:
                # if float(line[22:27].strip()) == start_ind:
                if float(line[22:26].strip()) == start_ind:
                    atoms.append(line)
                else:
                    start_ind = float(line[22:27].strip())
                    atoms.append(line)
                    pdb_lines.append('\n'.join(atoms))
                    index_resident.append(start_ind)
                    atoms = []
    pdb_lines.append('\n'.join(atoms))
    index_resident.append(start_ind)
    return pdb_lines, index_resident

def alignment_seq_add_(str1, str2):
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

aligner = Align.PairwiseAligner()
def alignment_score(str1, str2):
    """
    get similarity score
        :param str1: query sequence
        :param str2: input sequence
        :return: similarity score between 0 and 1
    """

    alignments = aligner.align(str1, str2)
    alignment = next(alignments)
    score = alignment.score

    # return score / max(len(str1), len(str2))
    return score / len(str2)

def seq_encoder(sequence, method="md5"):
    hasher = eval(f"hashlib.{method}")
    return hasher(sequence.encode(encoding="utf-8")).hexdigest()

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

def search_template_by_foldseek(query_pdb_dir, target_pdb_dir, output):
    
    if target_pdb_dir == 'gpcrDB':
        db_dir = os.path.join('/nfs_beijing/kubeflow-user/zhangyan_2023/code/protein_gap/foldseek_gpcr', target_pdb_dir)
        out_path = os.path.join('/nfs_beijing/kubeflow-user/zhangyan_2023/code/protein_gap/gpcr_result',output)
    elif target_pdb_dir == 'instrumentalDB':
        db_dir = os.path.join('/nfs_beijing/kubeflow-user/zhangyan_2023/code/protein_gap/foldseek_instrumental',target_pdb_dir)
        out_path = os.path.join('/nfs_beijing/kubeflow-user/zhangyan_2023/code/protein_gap/instrumental_result',output)
    else:
        raise 'target_pdb_dir wroung!!!'
    # os.popen(db_cmd)
    cmd = "/nfs_beijing/kubeflow-user/zhangyan_2023/code/protein_gap/foldseek/foldseek easy-search {} {} {} tmpFolder".format(query_pdb_dir, db_dir, out_path)
    with os.popen(cmd, "r") as p:
        result = p.read()
    
    return out_path


def image_compose(pic_path, pdb_name):

    IMAGES_PATH = os.path.join(pic_path, pdb_name)
    IMAGES_FORMAT = ['.jpg', '.JPG', '.png', '.PNG']  # 图片格式
    IMAGE_SIZE = 1000  # 每张小图片的大小
    IMAGE_ROW = 1  # 图片间隔，也就是合并成一张图后，一共有几行
    IMAGE_COLUMN = 3  # 图片间隔，也就是合并成一张图后，一共有几列
    IMAGE_SAVE_PATH = os.path.join(pic_path, f'{pdb_name}.png')

    # 获取图片集地址下的所有图片名称
    image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
                os.path.splitext(name)[1] == item]
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(os.path.join(IMAGES_PATH, image_names[IMAGE_COLUMN * (y - 1) + x - 1])).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    shutil.rmtree(IMAGES_PATH, ignore_errors=True)
    return to_image.save(IMAGE_SAVE_PATH)


def get_gap_pic(pdb_path, pdb_name, gap_list, save_path, is_tool, is_gpcr, save_pdb_path):
    chain = pdb_name[5]
    with open(pdb_path) as fp:
        pdb_str = fp.read()
    pdb_strs = ''
    for line in pdb_str.split('\n'):
        if line[:4] == 'ATOM':
            pdb_strs += line
            pdb_strs += '\n'


    smp_data = protein.from_pdb_string(
                            pdb_strs,
                            chain_id=chain,
                            use_filter_atom = True,
                            is_multimer= True,
                            return_id2seq = False,
                            is_cut_ab_fv = True,
                            resolution = 0.0,
                        )
    # make_pdb_path = '/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/test_pdb/'
    os.makedirs(save_pdb_path, exist_ok=True)
    make_pdb_path = os.path.join(save_pdb_path, pdb_name + '.pdb')
    with open(make_pdb_path, 'w') as fpw:
        fpw.write(protein.to_pdb(smp_data))
    
    cmd.load(make_pdb_path, pdb_name)
    cmd.bg_color(color="white")
    start_ind = int(gap_list[0][0])
    end_ind = int(gap_list[1][0])
    gap_list_new = []
    if len(gap_list[2]) > 1:
        for i in range(len(gap_list[2])):
            if int(gap_list[2][i]) <= 10:
                end_ind = int(gap_list[1][i])
            else:
                gap_list_new.append([start_ind, end_ind])
                start_ind = int(gap_list[0][i])
                end_ind = int(gap_list[1][i])
                
    gap_list_new.append([start_ind, end_ind])
    # is_tool_new.append(is_tool_token)
    # is_gpcr_new.append(is_gpcr_token)
    # print(gap_list_new)
    # 提取氨基酸的b-factor值
    
    myspace = {"lst":[]}
    cmd.iterate(f"name ca and model {pdb_name}",'lst.append(resi)', space = myspace)
    myspace = myspace['lst']
    resident_indexs = []
    print(len(myspace))
    
    pdb_indexs = np.sort(list(smp_data.residue_index))
    for resident_index in pdb_indexs:
        color_num = 2
        for ind, gap in enumerate(gap_list_new):
            if resident_index in range(gap[0],gap[1]+1):
                if is_tool != [] and is_tool[ind] == 1:
                    resident_indexs.append(0)
                elif is_gpcr != [] and is_gpcr[ind] == 1:
                    resident_indexs.append(1)
                else:
                    resident_indexs.append(color_num)
            if is_tool[ind] != 1 and is_gpcr[ind] != 1:
                color_num += 1
    print(len(resident_indexs))
    print()
    # 将b-factor值用于颜色映射
    # cmap = plt.get_cmap('brg')
    # norm = LogNorm(vmin=min(resident_indexs), vmax=max(resident_indexs))
    # colors = [norm(resident_index) for resident_index in resident_indexs]
    # print(colors)
    # colors_index = list(set(colors))
    # print(colors_index)
    # print(len(resident_indexs))

    # 标记颜色并保存图像文件
    # print(resident_indexs)
    # print(myspace)
    cmd.color('white', 'all')
    for i, color in enumerate(resident_indexs):
        cmd.color('0x' + utils.cnames[color][1:], f'resi {myspace[i]}')
        # cmd.color('0x' + utils.cnames[colors_index.index(color)][1:], f'resi {myspace[i]}')
    os.makedirs(os.path.join(save_path, pdb_name), exist_ok=True)
    # cmd.bg_color(color="white")
    cmd.png(os.path.join(save_path, pdb_name, f'{pdb_name}.png'), width=1000, height=1000, dpi=100)
    cmd.rotate('x', angle=90, selection = "all")
    cmd.png(os.path.join(save_path, pdb_name, f'{pdb_name}_x90.png'), width=1000, height=1000, dpi=100)
    cmd.rotate('y', angle=90, selection = "all")
    cmd.png(os.path.join(save_path, pdb_name, f'{pdb_name}_xy90.png'), width=1000, height=1000, dpi=100)

    # 关闭当前PDB文件
    cmd.delete('all')
    pic_path = save_path
    image_compose(pic_path, pdb_name)
    os.remove(make_pdb_path)

# pdb_path_dict = utils_zy.get_ag_seq_original_pdb_path()
pdb_path_dict = utils_zy.get_ag_pdb_path()
def process_epitope(np_exsamples, data_items):
    for row in data_items:

        try:
        # if True:
            is_tool = []
            is_gpcr = []
            # fname = row[0].strip()
            # fpath = pdb_path_dict[fname]
            # start = row[4].strip().split(';')
            # end = row[5].strip().split(';')
            # gap_len = row[6].strip().split(';')
            # tool_score = row[11].split(';')
            # tool_score2 = row[13].split(';')
            # print(tool_score)
            fname = row[1].strip()
            fpath = pdb_path_dict[fname]
            start = str(row[5]).strip().split(';')
            end = str(row[6]).strip().split(';')
            gap_len = str(row[7]).strip().split(';')
            tool_score = str(row[13]).split(';')
            tool_score2 = str(row[15]).split(';')
            for ind, i in enumerate(tool_score):
                if float(i) >= 0.5:
                    is_tool.append(1)
                else:
                    is_tool.append(0)
            for ind, i in enumerate(tool_score2):
                if float(i) >= 0.5:
                    is_gpcr.append(1)
                else:
                    is_gpcr.append(0)
            # save_path = '/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/img'
            save_path = '/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/test_img'
            # save_path = '/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/less30_img'
            
            
            save_pdb_path = '/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/test_pdb/'
            get_gap_pic(fpath, fname, [start, end, gap_len], save_path, is_tool, is_gpcr, save_pdb_path)
            
            
        except Exception as e:
            print(e)
            print('error!!!!!!!!!!!!!!!!!!!!!!!!!')
            break
            continue
        
        break


def start():
    # csv_path = '/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/data/data_profile_v10.csv'
    # csv_path = '/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/data/test.csv'
    # csv_path = '/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/data/gap.csv'
    # pdb_map = pd.read_csv(csv_path)
    excal_path = '/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/data/10416_max_gap_0_30.xlsx'
    pdb_map = pd.read_excel(excal_path)
    print(pdb_map)
    num_worker=1
    smps = np.array(pdb_map)
    print(smps[0])

    add = True
    np_exsamples = multi_process_wrapper(process_epitope, num_worker,smps[:])

if __name__ == '__main__':
    start()