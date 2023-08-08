from pathlib import Path

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

def test2():
    csv_path = '/nfs_beijing/kubeflow-user/zhangyan_2023/code/result/ag_ab_ap_auc.csv'
    df = pd.read_csv(csv_path)
    unrelax_AP = df['unrelax_AP']
    docking_AP = df['docking_AP']
    unrelax_AUC = df['unrelax_AUC']
    docking_AUC = df['docking_AUC']
    unrelax_Recall = df['unrelax_recall']
    docking_Recall = df['docking_recall']

    # plt.plot(range(len(unrelax_AP)), unrelax_AP, label='unrelax')
    # plt.plot(range(len(docking_AP)), docking_AP, label='docking')
    # plt.title("AP with unrelax and docking")
    # plt.xlabel("Data")
    # plt.ylabel("AP")
    # plt.legend()
    # plt.show()
    # plt.savefig('/nfs_beijing/kubeflow-user/zhangyan_2023/code/picture/ap.png')
    # plt.clf()
    # plt.plot(range(len(unrelax_AP)), sorted(unrelax_AP), label='unrelax')
    # plt.plot(range(len(docking_AP)), sorted(docking_AP), label='docking')
    # plt.title("AP(sort) with unrelax and docking")
    # plt.xlabel("Data")
    # plt.ylabel("AP")
    # plt.legend()
    # plt.show()
    # plt.savefig('/nfs_beijing/kubeflow-user/zhangyan_2023/code/picture/ap_sort.png')
    # plt.clf()
    # plt.plot(range(len(unrelax_AUC)), unrelax_AUC, label='unrelax')
    # plt.plot(range(len(docking_AUC)), docking_AUC, label='docking')
    # plt.title("AUC with unrelax and docking")
    # plt.xlabel("Data")
    # plt.ylabel("AUC")
    # plt.legend()
    # plt.show()
    # plt.savefig('/nfs_beijing/kubeflow-user/zhangyan_2023/code/picture/auc.png')
    # plt.clf()
    # plt.plot(range(len(unrelax_AUC)), sorted(unrelax_AUC), label='unrelax')
    # plt.plot(range(len(docking_AUC)), sorted(docking_AUC), label='docking')
    # plt.title("AUC(sort) with unrelax and docking")
    # plt.xlabel("Data")
    # plt.ylabel("AUC")
    # plt.legend()
    # plt.show()
    # plt.savefig('/nfs_beijing/kubeflow-user/zhangyan_2023/code/picture/auc_sort.png')
    # plt.clf()
    # plt.plot(range(len(unrelax_AUC)), unrelax_Recall, label='unrelax')
    # plt.plot(range(len(docking_AUC)), docking_Recall, label='docking')
    # plt.title("Recall with unrelax and docking")
    # plt.xlabel("Data")
    # plt.ylabel("Recall")
    # plt.legend()
    # plt.show()
    # plt.savefig('/nfs_beijing/kubeflow-user/zhangyan_2023/code/picture/recall.png')
    # plt.clf()
    # plt.plot(range(len(unrelax_AUC)), sorted(unrelax_Recall), label='unrelax')
    # plt.plot(range(len(docking_AUC)), sorted(docking_Recall), label='docking')
    # plt.title("Recall(sort) with unrelax and docking")
    # plt.xlabel("Data")
    # plt.ylabel("Recall")
    # plt.legend()
    # plt.show()
    # plt.savefig('/nfs_beijing/kubeflow-user/zhangyan_2023/code/picture/recall_sort.png')
    # plt.clf()
    print(np.mean(docking_Recall), np.mean(docking_AP), np.mean(docking_AUC))
    print(np.max(docking_Recall), np.max(docking_AP), np.max(docking_AUC))
    print(np.min(docking_Recall), np.min(docking_AP), np.min(docking_AUC))


if __name__ == '__main__':
    test2()