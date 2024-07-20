conda env create -f environment.yml
conda activate esmfold
conda install -c nvidia cuda-nvcc
conda install -c conda-forge einops=0.6.1

# install esmfold

pip install "fair-esm[esmfold]"

# OpenFold and its remaining dependency

pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
Here the host cuda version needs to correspond to the pytorch cuda version

pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

# install pytorch geometric and so on

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch-geometric

# install biotite

pip install biotite

install blast
cd esm/model
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.14.1+-x64-linux.tar.gz
wget https://ftp.ncbi.nlm.nih.gov/blast/db/FASTA/swissprot.gz
tar -zxvf ncbi-blast-2.14.1+-x64-linux.tar.gz
gzip -r swissprot.gz
./makeblastdb -in ../../swissprot -db type prot -title "swissprot" -out swissprot
install cd-hit
wget https://github.com/weizhongli/cdhit/releases/download/V4.6.7/cd-hit-v4.6.7-2017-0501-Linux-binary.tar.gz
tar -zxvf cd-hit-v4.6.7-2017-0501-Linux-binary.tar.gz
export PATH=./esm/model/ncbi-blast-2.14.1+/bin:$PATH
./cd-hit-v4.6.7-2017-0501/cd-hit -i db/uniprotkb.fasta -o out/uniprotkb40.fasta -c 0.4 -n 2 -T 10 -d 0
./cd-hit-v4.6.7-2017-0501/cd-hit-2d -i db/Phosphorylation-S-T-Y.fasta -i2 out/uniprotkb40.fasta -o out/uniprotkb40_2.fasta -c 0.4 -n 2 –T 10

The first step is to remove the duplicates of negative samples: the pos.fasta file here needs to enter its corresponding file path

> cd-hit -i neg.fasta -o pos40.fasta -c 0.4 -n 2

The second step is to perform deduplication on the positive and negative samples:

> cd-hit-2d -i pos.fasta -i2 neg.fasta -o neg40_1.fasta -c 0.4 -n 2

The third step is to get the negative samples and then perform the deduplication operation:

> cd-hit -i neg40_1.fasta -o neg40.fasta -c 0.4 -n 2
