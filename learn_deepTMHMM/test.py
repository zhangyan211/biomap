import biolib
import os

# print(os.environ.get('DOCKER_CERT_PATH'))
deeptmhmm = biolib.load('DTU/DeepTMHMM')
# biolib.utils.BIOLIB_DOCKER_RUNTIME = 'nvidia'
biolib.utils.STREAM_STDOUT = True # Stream progress from app in real time
deeptmhmm_job = deeptmhmm.cli(args=
                              '--fasta /nfs_beijing/kubeflow-user/zhangyan_2023/code/learn_deepTMHMM/rcsb_pdb_2B4C.fasta',
                              machine='local') # Blocks until done
deeptmhmm_job.save_files('/nfs_beijing/kubeflow-user/zhangyan_2023/code/learn_deepTMHMM/result.txt')