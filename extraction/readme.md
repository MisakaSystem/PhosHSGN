This folder is mainly used to run the protein feature processing backend. Integrated esm protran seqvec for feature extraction for protein embeddings.

# install
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

#install blast
cd esm/model
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.14.1+-x64-linux.tar.gz
wget https://ftp.ncbi.nlm.nih.gov/blast/db/FASTA/swissprot.gz
tar -zxvf ncbi-blast-2.14.1+-x64-linux.tar.gz
gzip -r swissprot.gz
./makeblastdb -in ../../swissprot -db type prot -title "swissprot" -out swissprot
export PATH=./esm/model/ncbi-blast-2.14.1+/bin:$PATH

# run
python esm_api.py 
