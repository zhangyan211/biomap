import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm
from pymol import cmd
import utils


color_name_list = utils.cnames
# print(color_name_list)
def BGR2BGR_hex(bgr_tuple):
    assert isinstance(bgr_tuple, tuple)
    assert len(bgr_tuple) == 3
    b = int(bgr_tuple[0] * 255)
    g = int(bgr_tuple[1] * 255)
    r = int(bgr_tuple[2] * 255)
    bgr_hex = hex(b*16**4 + g*16**2 + r)
    return bgr_hex

def get_pic_all():
    pdb_list = '/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/test_pdb'
    for filename in os.listdir(pdb_list):
        if filename.endswith('.pdb'):
            # 读取PDB文件
            cmd.load(os.path.join(pdb_list, filename))

            # 提取氨基酸的b-factor值
            b_factors = []
            with open(os.path.join(pdb_list, filename)) as f:
                for line in f:
                    if line.startswith('ATOM'):
                        b_factor = float(line[60:66])
                        b_factors.append(b_factor)

            # 将b-factor值用于颜色映射
            cmap = plt.get_cmap('hsv')
            norm = LogNorm(vmin=min(b_factors), vmax=max(b_factors))
            # norm = plt.Normalize(vmin=min(b_factors), vmax=max(b_factors))
            colors = [cmap(norm(b_factor)) for b_factor in b_factors]
            colors_index = list(set(colors))
            print(len(colors_index))

            # 标记颜色并保存图像文件
            cmd.color('white', 'all')
            for i, color in enumerate(colors):
                cmd.color(BGR2BGR_hex(color[:-1]), f'resi {i+1}')
            cmd.png(os.path.join('/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/test_img', f'{filename}.png'), width=800, height=600, dpi=300)

            # 关闭当前PDB文件
            cmd.delete('all')

def test_gap_pic(filename, gap_list):
    pdb_list = '/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/test_pdb'
    pdb_name = filename.split('.')[0]
    cmd.load(os.path.join(pdb_list, filename), pdb_name)
    
    gap_list = [[gap_list[0][i], gap_list[1][i]] for i in range(len(gap_list[0]))]
    
    # 提取氨基酸的b-factor值
    
    myspace = {"lst":[]}
    cmd.iterate(f"name ca and model {pdb_name}",'lst.append(resi)', space = myspace)
    myspace = myspace['lst']
    resident_indexs = []
    resident_index_start = -1
    with open(os.path.join(pdb_list, filename)) as f:
        for line in f:
            if line.startswith('ATOM'):
                resident_index = line[22:27]
                if resident_index[-1].isalpha():
                    resident_index = resident_index[:-1]
                    # continue
                resident_index = int(resident_index)
                if resident_index_start != resident_index:
                    resident_index_start = resident_index
                    color_num = 1
                    for gap in gap_list:
                        if resident_index in range(gap[0],gap[1]+1):
                            resident_indexs.append(color_num)
                        color_num += 100

    # 将b-factor值用于颜色映射
    cmap = plt.get_cmap('brg')
    norm = LogNorm(vmin=min(resident_indexs), vmax=max(resident_indexs))
    # norm = plt.Normalize(vmin=min(b_factors), vmax=max(b_factors))
    colors = [norm(resident_index) for resident_index in resident_indexs]
    colors_index = list(set(colors))
    # print(colors_index)
    print(len(colors))

    # 标记颜色并保存图像文件
    cmd.color('white', 'all')
    for i, color in enumerate(colors):
        cmd.color('0x' + color_name_list[colors_index.index(color)][1:], f'resi {myspace[i]}')
    cmd.png(os.path.join('/nfs_beijing/kubeflow-user/zhangyan_2023/code/py3Dmol/test_img', f'{pdb_name}.png'), width=4000, height=4000, dpi=100)

    # 关闭当前PDB文件
    cmd.delete('all')
    
    
if __name__ =='__main__':
    # test_gap_pic('6os0_A.pdb', [[9,323],[1226,1317]])
    # test_gap_pic('7ugp_C.pdb', [[32,64,152,186,312,321,411],[57,138,186,309,321,397,506]])
    # test_gap_pic('7wxw_A.pdb', [[8,81,264],[64,253,394]])
    # test_gap_pic('8hcx_C.pdb', [[90,200],[200,397]])
    get_pic_all()