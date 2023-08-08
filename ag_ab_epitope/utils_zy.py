def get_all_original_seq():
    fasta_path = '/nfs_baoding/share/shaochuan/seq/pdb_seqres.fasta'
    original_seq = {}
    with open(fasta_path) as fp:
        for line in fp:
            name = line.strip().split()[0][1:]
            name = name.split('_')[0].upper() + '_' + name.split('_')[1]
            seq = fp.readline().strip()
            original_seq[name] = seq
    return original_seq


if __name__ =='__main__':
    get_all_original_seq()