import torch.nn
from dgl.nn.pytorch import GraphConv, AvgPooling, MaxPooling
from torch import nn
import torch.nn.functional as F
from layer import *
class LuGTP(torch.nn.Module):
    def __init__(self, config,inputfeature=1280, hiddsize=128):
        super(LuGTP, self).__init__()
        if config.emb_type=='esm':
            self.inputfeature = 1280
        if config.emb_type=='prot' or config.emb_type=='seqvec':
            self.inputfeature = 1024
        self.hiddsize = hiddsize
        self.gconv1 = GraphConv(self.inputfeature, hiddsize*3, weight=True, bias=True)
        self.gconv2 = GraphConv(hiddsize*3,hiddsize*2, weight=True, bias=True)
        self.gconv3 = GraphConv(hiddsize*2, hiddsize, weight=True, bias=True)
        self.relu = nn.ReLU()
        self.seq1 = nn.Linear(128, 32)
        self.seq2 = nn.Linear(32, 2)
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        self.w0 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.w1 = nn.Parameter(torch.FloatTensor([0.33]), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor([0.33]), requires_grad=True)
        self.w3 = nn.Parameter(torch.FloatTensor([0.33]), requires_grad=True)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(6528+1536, 128)
        self.seqphosidn=Phosidn(config)
    def forward(self, gbatch, pad_dmap, seqmatrix, pssms, dssps, concatdata):
        readout0 = self.seqphosidn(concatdata)
        # First convolution + pooling
        g1 = self.gconv1(gbatch, gbatch.ndata['feat'])
        g1 = self.relu(g1)
        gbatch.ndata['feat'] = g1
        G=gbatch
        x=g1
        avx = self.avgpool(G, x)
        maxx = self.maxpool(G, x)
        readout1 = torch.cat([avx, maxx], dim=1)
        # Second convolution + pooling
        g2 = self.gconv2(G, G.ndata['feat'])
        g2 = self.relu(g2)
        G.ndata['feat'] = g2

        G2 = G
        x2 = g2
        avx2 = self.avgpool(G2, x2)
        maxx2 = self.maxpool(G2, x2)
        readout2 = torch.cat([avx2, maxx2], dim=1)

        # The third convolution + pooling
        g3 = self.gconv3(G2, G2.ndata['feat'])
        g3 = self.relu(g3)
        G2.ndata['feat'] = g3
        G3 = G2
        x3 = g3
        avx3 = self.avgpool(G3, x3)
        maxx3 = self.maxpool(G3, x3)
        readout3 = torch.cat([avx3, maxx3], dim=1)

        readoutcat=torch.concat([readout1,readout2,readout3],dim=1)
        readout= self.relu(readoutcat)
        gnn=readout

        readout0 = self.flat(readout0)
        readout0 = self.relu(readout0)
        readout0 = readout0.unsqueeze(1)
        gnn=gnn.unsqueeze(1)
        feat=torch.cat([readout0,gnn],dim=-1)
        feat=feat.squeeze(1)
        feat = self.fc(feat)
        feat = self.relu(feat)
        seq = self.seq1(feat)
        seq=self.relu(seq)
        seq= self.seq2(seq)
        seq=self.relu(seq)
        output = torch.softmax(seq, dim=1)
        return output
        readout0 = readout0.unsqueeze(1)
        gnn=gnn.unsqueeze(1)
        # feat=torch.cat([readout0,gnn],dim=-1)
        feat=gnn.squeeze(1)
        feat = self.fc(feat)
        feat = self.relu(feat)
        seq = self.seq1(feat)
        seq=self.relu(seq)
        seq= self.seq2(seq)
        seq=self.relu(seq)
        output = torch.softmax(seq, dim=1)
        return output

