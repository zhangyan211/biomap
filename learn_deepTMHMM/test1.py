import biolib

deeptmhmm = biolib.load('DTU/DeepTMHMM')
# biolib.utils.BIOLIB_DOCKER_RUNTIME = 'nvidia'
biolib.utils.STREAM_STDOUT = True # Stream progress from app in real time
deeptmhmm_job = deeptmhmm.cli(args=
                              '--fasta /nfs_beijing/kubeflow-user/zhangyan_2023/code/learn_deepTMHMM/rcsb_pdb_2B4C.fasta') # Blocks until done
deeptmhmm_job.save_files('/nfs_beijing/kubeflow-user/zhangyan_2023/code/learn_deepTMHMM/result2.txt')