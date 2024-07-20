from random import sample

import dgl
import numpy as np
import torch
from dgl.nn.pytorch import GraphConv, GATv2Conv, AvgPooling, MaxPooling
from torch import nn, topk, tensor

class Lupool(nn.Module):
    def __init__(self, in_channels, ratio=0.7):
        super(Lupool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.relu = nn.ReLU()
        self.non_linearity = torch.tanh
        self.hid = int(in_channels / 2)
        self.score_layer = GraphConv(in_channels, in_channels+self.hid ,weight=True, bias=True)
        # self.score_layer2 = GATv2Conv(in_channels+self.hid, in_channels, num_heads=1)
        self.score_sum = GraphConv(in_channels+self.hid, 1, weight=True, bias=True)

    def forward(self, G, x):
        score = self.score_layer(G, x)
        score = self.relu(score)
        # g1 = score.reshape(-1, self.in_channels+self.hid)
        # score2 = self.score_layer2(G, g1)
        # score2 = self.relu(score2)
        # g2 = score2.reshape(-1, self.in_channels)
        socres = self.score_sum(G, score).squeeze()
        nodesid = G.nodes()
        npnodelist = nodesid.cpu().numpy()
        value, indices = topk(torch.abs(socres), int(len(nodesid) * self.ratio), largest=True)
        npindices = indices.cpu().numpy()
        diff = np.setdiff1d(npnodelist, npindices)
        # x=x[indices]*self.non_linearity(socres[indices]).view(-1, 1)
        G.remove_nodes(diff)
        return G, G.ndata['feat']
class Self_Attention(nn.Module):
    def __init__(self,input,output):
        super(Self_Attention, self).__init__()
        self.output_dim=output
        w= torch.rand(input, output)
        self.WQ = nn.Parameter(torch.FloatTensor(w), requires_grad=True)
        # nn.init.kaiming_normal_(self.WQ, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.xavier_uniform_(self.WQ)
        self.WK = nn.Parameter(torch.FloatTensor(w), requires_grad=True)
        # nn.init.kaiming_normal_(self.WK, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.xavier_uniform_(self.WK)
        self.WV = nn.Parameter(torch.FloatTensor(w), requires_grad=True)
        # nn.init.kaiming_normal_(self.WV, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.xavier_uniform_(self.WV)

    def forward(self,x):
        x=x.permute(0, 2, 1)
        WQ=x.matmul(self.WQ)
        WK = x.matmul(self.WK)
        WV = x.matmul(self.WV)
        T = WK.permute(0, 2, 1)
        QK=WQ.matmul(T)
        QK=QK/(self.output_dim**0.5)
        V=QK.matmul(WV)
        return V

class convlay(nn.Module):
    def __init__(self,dropratio,input,hidd):
        super(convlay, self).__init__()
        self.dropratio = dropratio
        self.hidd=hidd
        self.input=input
        self.conv1 = nn.Conv1d(self.input, self.hidd, kernel_size=7,padding=3)
        self.relu = nn.ReLU()
        self.drop=nn.Dropout()
    def forward(self,x):
        x = self.conv1(x)
        x=self.relu(x)
        x = self.drop(x)
        return x
class Denseblock(nn.Module):
    def __init__(self,blocknum,dropratio,input,grow_size,config):
        super(Denseblock, self).__init__()
        self.blocknum=blocknum
        self.dropratio=dropratio
        self.hidd=grow_size
        self.input=input
        self.conv=[]
        self.grow_size=grow_size
        self.temp=self.input
        for i in range(blocknum):
            self.conv.append(convlay(dropratio=dropratio,input=self.temp,hidd=self.grow_size).to(config.device))
            self.temp=self.temp+self.grow_size
    def forward(self,x):
        list_feat=[x]
        for i in range(self.blocknum):
            x=self.conv[i](x)
            list_feat.append(x)
            x=torch.cat(list_feat,dim=1)
        return x

class Phosidn(nn.Module):
    def __init__(self,config):
        super(Phosidn, self).__init__()
        self.conv1=nn.Conv1d(50,64,kernel_size=1)
        self.relu=nn.ReLU()
        self.dense=Denseblock(5,0.7,64,24,config)
        self.attention=Self_Attention(184,128)
        self.drop=nn.Dropout(config.dropout_ratio)
    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.dense(x)
        x=self.attention(x)
        x=self.relu(x)
        x=self.drop(x)

        return x

class ConvseqLayer(torch.nn.Module):
    def __init__(self,inputfeature):
        super(ConvseqLayer, self).__init__()
        self.inputfeature=inputfeature
        self.hiddlayer=128
        self.conv1 = nn.Conv1d(in_channels=self.inputfeature, out_channels=128, kernel_size=3)
        self.resv = ResLayer(self.hiddlayer)
        self.relu=nn.ReLU()
        self.mx1 = nn.MaxPool1d(3, stride=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        self.mx2 = nn.MaxPool1d(3, stride=3)
        self.mx3 = nn.MaxPool1d(4, stride=1)

    def forward(self,x):
        x = x.squeeze(1)
        x = x.permute(0,2,1)
        x1=self.conv1(x)
        x1 = self.mx1(x1)
        x1=self.relu(x1)
        x1=self.conv2(x1)
        x1=self.mx2(x1)
        x1=self.resv(x1)
        x1=self.mx3(x1)
        x1 = x1.squeeze(2)
        return x1
class ResLayer(torch.nn.Module):
    def __init__(self, emb_dim):
        super(ResLayer, self).__init__()
        self.embedding_size = emb_dim
        self.conv = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.embedding_size, kernel_size=1)
        self.relu=nn.ReLU()
        self.norm = nn.BatchNorm1d(self.embedding_size)
        self.drop=nn.Dropout()
    def forward(self, x):
        features = self.conv(x)
        xx=features+x
        xx=self.norm(xx)
        xxx=self.relu(xx)
        xxx=self.drop(xxx)
        xxxx=self.conv(xxx)
        out=self.norm(self.relu(xxxx+features))
        return out

class ConvsLayer(torch.nn.Module):

    def __init__(self, emb_dim):
        super(ConvsLayer, self).__init__()
        self.embedding_size = emb_dim
        self.hiddlayer=128
        self.conv1 = nn.Conv1d(in_channels=self.embedding_size, out_channels=self.hiddlayer, kernel_size=3)
        self.mx1 = nn.MaxPool1d(3, stride=3)
        self.conv2 = nn.Conv1d(in_channels=self.hiddlayer,  out_channels=self.hiddlayer, kernel_size=3)
        self.resv2=ResLayer(self.hiddlayer)
        self.mx2 = nn.MaxPool1d(3, stride=3)
        self.conv3 = nn.Conv1d(in_channels=self.hiddlayer, out_channels=self.hiddlayer, kernel_size=3)
        self.resv3 = ResLayer(self.hiddlayer)
        self.mx3 = nn.MaxPool1d(3, stride=3)
        self.conv4 = nn.Conv1d(in_channels=self.hiddlayer, out_channels=self.hiddlayer, kernel_size=3)
        self.resv4 = ResLayer(self.hiddlayer)
        self.mx4 = nn.MaxPool1d(3, stride=3)
        self.conv5 = nn.Conv1d(in_channels=self.hiddlayer, out_channels=self.hiddlayer, kernel_size=3)

        self.mx5 = nn.MaxPool1d(10, stride=1)
        self.norm=nn.BatchNorm1d(self.hiddlayer)
        self.flat = nn.Flatten()
    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        features = self.conv1(x)
        features = self.mx1(features)
        features=self.resv2(features)
        features = self.mx2(features)
        features = self.resv3(features)
        features = self.mx3(features)
        features = self.resv3(features)
        features = self.mx4(features)
        features=self.flat(features)
        return features