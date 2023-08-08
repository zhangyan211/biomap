import os

from tqdm import tqdm

import numpy as np
import pandas as pd
import pymysql
from Bio import Align

import torch

import requests
from bs4 import BeautifulSoup


aligner = Align.PairwiseAligner()

conn = pymysql.connect(
    host = '172.16.0.99',  # 远程主机的ip地址， 
    user = 'u_biomap_yiwu',   # MySQL用户名
    passwd = 'Yiwu@230620!',   # 数据库密码
    port = 9030,  #数据库监听端口，默认3306
    charset = "utf8"
)  #指定utf8编码的连接

cursor = conn.cursor()

def getCode(url, encoding='UTF-8'):
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
    response = requests.get(url, headers=headers)
    response.encoding = encoding
    return BeautifulSoup(response.text, 'html.parser')

def find_ecd_with_uniprot_id(cursor, uniprot_id):
    try:
        sql = f"select * from pyuniprot.pyuniprot_accession WHERE f_accession='{uniprot_id}'"
        cursor.execute(sql)
        values = cursor.fetchall()
        if not values:
            return None, None
        f_entry_id = values[0][-1]
    except:
        conn = pymysql.connect(
            host = '172.16.0.99',  # 远程主机的ip地址， 
            user = 'u_biomap_yiwu',   # MySQL用户名
            passwd = 'Yiwu@230620!',   # 数据库密码
            port = 9030,  #数据库监听端口，默认3306
            charset = "utf8"
        )  #指定utf8编码的连接

        cursor = conn.cursor()
        sql = f"select * from pyuniprot.pyuniprot_accession WHERE f_accession='{uniprot_id}'"
        cursor.execute(sql)
        values = cursor.fetchall()
        if not values:
            return None, None
        f_entry_id = values[0][-1]

    sql = f"select * from pyuniprot.pyuniprot_sequence WHERE f_entry_id={f_entry_id}"
    cursor.execute(sql)
    values = cursor.fetchall()
    uniprot_seq = values[0][1]

    # sql = f"select * from pyuniprot.pyuniprot_feature WHERE f_entry_id={f_entry_id} AND f_type_='topological domain' AND f_description='Extracellular'"
    sql = f"select * from pyuniprot.pyuniprot_feature WHERE f_entry_id={f_entry_id}"
    cursor.execute(sql)
    values = cursor.fetchall()
    
    if values:
        data = pd.DataFrame(
            list(values),
            columns=[
                'f_id',
                'f_type_',
                'f_identifier',
                'f_description',
                'f_f_begion',
                'f_f_end',
                'f_f_position',
                'f_entry_id',
            ]
        )
    else:
        return None, None
    return data, uniprot_seq

    
def parse_uniprot_id(pdb_id, ag_chain):
    url = f"https://www.rcsb.org/structure/{pdb_id.upper()}"
    wb = getCode(url)
    
    chain_found = False
    tables = wb.find_all('div', class_='table-responsive')
    for table in tables:
        trs = table.find_all('tr')
        for tr in trs:
            tds = tr.find_all('td')
            if tds:
                for td in tds:
                    tmp = td.find_all('a')
                    for a in tmp:
                        if a.attrs['href'][-1] == ag_chain:
                            chain_found = True
                            break
                    if chain_found:
                        divs = td.find_all('div')
                        for div in divs:
                            text = div.get_text().strip()
                            if "Go to UniProtKB" in text:
                                return text.split(' ')[-1].strip()
    
    sub_choice = f"[auth {ag_chain}]"
    for table in tables:
        trs = table.find_all('tr')
        for tr in trs:
            tds = tr.find_all('td')
            if tds:
                for td in tds:
                    tmp = td.find_all('a')
                    for a in tmp:
                        if sub_choice in a.get_text().strip():
                            chain_found = True
                            break
                    if chain_found:
                        divs = td.find_all('div')
                        for div in divs:
                            text = div.get_text().strip()
                            if "Go to UniProtKB" in text:
                                return text.split(' ')[-1].strip()
    
    return None

# feat_root = '/nfs_beijing/kubeflow-user/zhangyan_2023/code'

# train_df = pd.read_csv('/pfs_beijing/share/sunyiwu/training_files/ab_ag_training_v7.csv')
if __name__ == '__main__':
    train_df = pd.read_csv('/nfs_baoding_ai/shaochuan/code/ds_check/ab_ag_training_v6.csv')
    ag_uniprots = []
    for i, row in tqdm(train_df[len(ag_uniprots):].iterrows()):
        pdb_file_path = row.pdb_file_path
        pdb = row.pdb_file_path.split('/')[-1][:-4]
        pdb_id = pdb[-4:]
        chains = row.chain.replace(',', '')
        
        ag_chain = chains[-1]

        # fasta_file = os.path.join(feat_root, 'hu', f"{pdb}_{chains}", "sequences.fasta")
        # with open(fasta_file, 'r') as f:
        #     ag_seq = f.read().splitlines()[-1]

        try:
            uniprot_id = parse_uniprot_id(pdb_id, ag_chain)
            ag_uniprots.append(uniprot_id)
        except:
            print(pdb, ag_chain, uniprot_id)
            raise KeyError
        break

    print(ag_uniprots)