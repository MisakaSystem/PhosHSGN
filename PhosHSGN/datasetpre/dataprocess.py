import csv
import zlib
import sys
import dgl
import scipy.sparse as spp
import torch
from dgl import save_graphs, load_graphs
from datasetpre.http_emb import get_emb, get_protemb, get_seqvecemb
from datasetpre.http_cmap import get_cmap
from datasetpre.http_pssm import get_pssm
from datasetpre.http_dssp import get_dssp
from datasetpre.pssm_compute import *
from datasetpre.dssp_compute import *

cmap_path='data/cmap/'
letterDict = {}
letterDict["A"] = 0
letterDict["C"] = 1
letterDict["D"] = 2
letterDict["E"] = 3
letterDict["F"] = 4
letterDict["G"] = 5
letterDict["H"] = 6
letterDict["I"] = 7
letterDict["K"] = 8
letterDict["L"] = 9
letterDict["M"] = 10
letterDict["N"] = 11
letterDict["P"] = 12
letterDict["Q"] = 13
letterDict["R"] = 14
letterDict["S"] = 15
letterDict["T"] = 16
letterDict["V"] = 17
letterDict["W"] = 18
letterDict["Y"] = 19
letterDict["*"] = 20
letterDict["X"] = 20

def listclass_to_one_hot(list, isnumpy=True):
    list_len = len(list)
    li = []
    for i in range(list_len):
        li.append([list[i]])
    one_hot_list = torch.LongTensor(li)
    # label one-hot encoding format.
    if isnumpy:
        one_hot_list = torch.zeros(list_len, 2).scatter_(1, one_hot_list, 1).numpy()
    else:
        one_hot_list = torch.zeros(list_len, 2).scatter_(1, one_hot_list, 1)
    return one_hot_list
def seq_load(file_path):
    seqs=[]
    proteins=[]
    labels=[]
    positions=[]
    with open(file_path, 'r') as rf:
        reader = csv.reader(rf)
        next(reader)
        for row in reader:
            sseq = row[3]
            protein = row[1]
            position = int(row[2])
            label=int(row[0])
            seqs.append(sseq)
            proteins.append(protein)
            labels.append(label)
            positions.append(position)
    return proteins,seqs,labels,positions
def getdssps(seqs,proteins,positions,window_size=51, empty_aa="*"):
    dssps=[]
    processlength = len(seqs)
    half_len = (window_size - 1) / 2
    for index, (seq, protein, position) in enumerate(zip(seqs, proteins, positions)):
        processper = int(index / processlength * 100)
        progress_bar = "▉" * int(processper/10)
        sys.stdout.write(f"\rseqdssp:{progress_bar} {processper}%  ")
        sys.stdout.flush()
        dssp_path=get_dssp(seq,protein)
        dssp = loaddsspfile(dssp_path, protein ,position, len(seq))
        dssps.append(dssp)
    return dssps
def getpssms(seqs,proteins,positions,window_size=51, empty_aa="*"):
    pssms=[]
    processlength = len(seqs)
    half_len = (window_size - 1) / 2
    for index, (seq, protein, position) in enumerate(zip(seqs,proteins,positions)):
        processper = int(index / processlength * 100)
        progress_bar = "▉" * int(processper/10)
        sys.stdout.write(f"\rseqpssm:{progress_bar} {processper}%  ")
        sys.stdout.flush()
        if position < (window_size - 1) / 2:
            start = 0
            l_padding = (window_size - 1) / 2 - position
        else:
            start = position - (window_size - 1) / 2
            l_padding = 0
        if position > len(seq) - (window_size - 1) / 2:
            end = len(seq)
            r_padding = (window_size - 1) / 2 - (len(seq) - position)
        else:
            end = position + (window_size - 1) / 2
            r_padding = 0
        pssm_path=get_pssm(seq,protein)
        pssm = load_fasta_and_compute(protein, position, pssm_path, start, end, l_padding, r_padding)
        pssms.append(np.transpose(pssm))
    return pssms
def getseqmatrix(seqs,proteins,positions,labels,window_size=51, empty_aa='*'):
    processlength = len(seqs)
    half_len = (window_size - 1) / 2
    short_seqs=[]
    for index,(seq,protein,position,label) in enumerate(zip(seqs,proteins, positions,labels)):
        processper = int(index / processlength * 100)
        progress_bar = "▉" * int(processper/10)  # progress
        sys.stdout.write(f"\rseqmatrix:{progress_bar} {processper}%  ")
        sys.stdout.flush()
        try:
            center = seq[position - 1]
        except BaseException:
            seqlength=len(seq)
            print(f"error protein {protein} pos {position} len {seqlength}")
        # short seq
        if position - half_len > 0:
            start = position - int(half_len)
            left_seq = seq[start - 1:position - 1]
        else:
            left_seq = seq[0:position - 1]

        end = len(seq)
        if position + half_len < end:
            end = position + half_len
        right_seq = seq[position:int(end)]

        if len(left_seq) < half_len:
            nb_lack = half_len - len(left_seq)
            left_seq = ''.join([empty_aa for count in range(int(nb_lack))]) + left_seq

        if len(right_seq) < half_len:
            nb_lack = half_len - len(right_seq)
            right_seq = right_seq + ''.join([empty_aa for count in range(int(nb_lack))])
        shortseq = left_seq + center + right_seq
        short_seqs.append(shortseq)
    targetY = listclass_to_one_hot(labels,isnumpy=False)
    ONE_HOT_SIZE = 21
    # _aminos = 'ACDEFGHIKLMNPQRSTVWY*'
    Matr = np.zeros((len(short_seqs), window_size, ONE_HOT_SIZE))
    samplenumber = 0
    for sq in short_seqs:
        # print(seq)
        AANo = 0
        for AA in sq:
            index = letterDict[AA]
            Matr[samplenumber][AANo][index] = 1
            AANo = AANo + 1
        samplenumber = samplenumber + 1

    return Matr, targetY, proteins, positions
def get_graph(sseqs,proteins,arg_config):
    if arg_config.emb_type == 'esm':
        graph_path = 'data/graph/'
        emb_path = 'data/emb/'
        emblength=1280
    if arg_config.emb_type == 'prot':
        graph_path = 'data/protgraph/'
        emb_path = 'data/protemb/'
        emblength = 1024
    if arg_config.emb_type == 'seqvec':
        graph_path = 'data/seqvecgraph/'
        emb_path = 'data/seqvecemb/'
        emblength = 1024
    processlength=len(sseqs)
    glist=[]
    emblist=[]

    for index,(seq,protein) in enumerate(zip(sseqs,proteins)):
        processper = int(index / processlength * 100)
        progress_bar = "▉" * int(processper/10)
        sys.stdout.write(f"\rgraph:{progress_bar} {processper}%  ")
        sys.stdout.flush()

        graphtag = os.path.exists(graph_path + protein + ".g")
        # Check if the graph exists
        if graphtag == False:
            # Check if cmap exists
            cmap_tag = os.path.exists(cmap_path + protein + ".npy")
            if cmap_tag == False:
                cmdata = get_cmap(seq,protein)
                np.save(cmap_path + protein + ".npy", cmdata)
            else:
                try:
                    cmdata = np.load(cmap_path + protein + ".npy")
                except ValueError:
                    print(f"[debug]:{cmap_path} + {protein} + '.npy' error")
            # Check if emd exists
            emd_tag = os.path.exists(emb_path + protein + ".npy")
            if emd_tag == False:
                if arg_config.emb_type == 'esm':
                    g_embed = get_emb(seq,protein)

                if arg_config.emb_type == 'prot':
                    g_embed = get_protemb(seq,protein)
                if arg_config.emb_type == 'seqvec':
                    g_embed = get_seqvecemb(seq, protein)
                np.save(emb_path + protein + ".npy", g_embed)
            else:
                dat = np.load(emb_path + protein + ".npy")
                g_embed = dat
            g_embed = torch.from_numpy(g_embed)
            try:
                if len(g_embed) != len(cmdata):
                    if len(g_embed) < len(cmdata):
                        cmdata = cmdata[:len(g_embed), :len(g_embed)]
                    if len(g_embed) > len(cmdata):
                        g_embed = g_embed[:len(cmdata)]
            except TypeError:
                print("debug")
            adj = spp.coo_matrix(cmdata)
            G = dgl.from_scipy(adj)
            try:
                G.ndata['feat'] = g_embed.float()
            except:
                print("debug")
            graph_labels = {"glabel": torch.tensor([])}
            save_graphs(graph_path + protein + ".g", [G], graph_labels)
            g=G
            glist.append(G)
            edgindex = g.edges()
            g_embed = g.ndata['feat']
            nodenum = len(g_embed)
            if nodenum >= arg_config.cutproteinlen:
                textembed = g_embed[:arg_config.cutproteinlen]
            elif nodenum < arg_config.cutproteinlen:
                textembed = np.concatenate((g_embed, np.zeros((arg_config.cutproteinlen - nodenum, emblength))))
                textembed = torch.from_numpy(textembed)
            emblist.append(textembed)
        else:
            try:
                G, label = load_graphs(graph_path + protein + ".g")
            except BaseException:
                print("debug")
            # labels.append(label)
            g = G[0]
            glist.append(g)
            edgindex = g.edges()
            try:
                g_embed = g.ndata['feat']
            except KeyError:
                print("debug")
            nodenum = len(g_embed)
            if nodenum >= arg_config.cutproteinlen:
                textembed = g_embed[:arg_config.cutproteinlen]
            elif nodenum < arg_config.cutproteinlen:
                textembed = np.concatenate((g_embed, np.zeros((arg_config.cutproteinlen - nodenum, emblength))))
                textembed = torch.from_numpy(textembed)
            emblist.append(textembed)
    return glist,emblist


