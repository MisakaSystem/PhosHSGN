fasta_path="D:\迅雷下载\CDK_neg.fasta"
seqs = dict()
with open(fasta_path, 'r') as fasta_f:
    for line in fasta_f:
        # get uniprot ID from header and create new entry
        if line.startswith('>'):
            protein_id = line.replace('>', '')
            # # replace tokens that are mis-interpreted when loading h5
            # uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
            # protein_id = uniprot_id.split("|")[1]
            seqs[protein_id] = ''
        else:
            # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
            seq = ''.join(line.split()).upper().replace("-", "")
            # repl. all non-standard AAs and map them to unknown/X
            seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
            seqs[protein_id] += seq
y_count=0
s_count=0
t_count=0
for data in iter(seqs):
    seq = seqs[data]
    print(seq[10])

    if(seq[10]=='Y'):
        y_count=y_count+1
    if (seq[10] == 'S'):
        s_count = s_count + 1
        print(seq[10])
    if (seq[10] == 'T'):
        t_count = t_count + 1
        print(seq[10])
print(f"y:{y_count}")
print(f"s:{s_count}")
print(f"t:{t_count}")