import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp
from models.MolTrans import MTs_config, Moltrans
from models.ProVAE import CNN,decoder
from models.PairMap import BiAttention
from models.ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
import math
import numpy as np

MTconfig = MTs_config()


# GAT  model
class DrugMGR(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(DrugMGR, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = Linear(output_dim, 100 * output_dim)


        # substructure layer
        self.moltrans = Moltrans(**MTconfig)

        #drug sequence layer
        self.embedding_xd = nn.Embedding(64, embed_dim)
        self.conv_xd1 = nn.Conv1d(in_channels=100, out_channels=n_filters, kernel_size=7, padding = 3)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.ProCNN = CNN(32,7)
        self.fc_xt1 = Linear(32*121, output_dim)

        #Re-Construct Protein
        self.ReCons_Prot = decoder(1000, 32, 7, 26)

        self.bcn = weight_norm(
            BANLayer(v_dim=128, q_dim=128, h_dim=128, h_out=1),
            name='h_mat', dim=None)


        # combined layers
        self.fc1 = Linear(384, 1024)
        self.fc2 = Linear(1024, 256)
        self.out = Linear(256, n_output)

        self.gcs_attention = BiAttention(3*embed_dim,embed_dim,4)


        # activation and regularization
        self.relu = ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch

        print("type:", x.shape)

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)

        x = self.relu(x)
        print("x:", x.shape)
        x = gmp(x, batch)          # global max pooling
        print("gmp:", x.shape)
        x = self.relu(x)
        x = self.fc_g1(x)
        x = x.view(-1, 100, 128)
        print("graphx:", x.shape)



        d_v, d_mask = data.d_v, data.d_mask
        sub_s = self.moltrans(d_v,d_mask)
        print(sub_s.shape)

        #sequence
        seq_xd = data.seq_xd
        embedded_xd = self.embedding_xd(seq_xd)
        conv_xd = self.conv_xd1(embedded_xd)
        conv_xd = self.relu(conv_xd)
        print("conv_xd:", conv_xd.shape)


        # protein input feed-forward:
        target = data.target
        embedded_xt = self.embedding_xt(target)

        embedded_xt = embedded_xt.permute(0,2,1)

        out_t,mu_t,logvar_t = self.ProCNN(embedded_xt)
        out_t = out_t.permute(0, 2, 1)
        print("Target:", out_t.shape)
        recons_protein = self.ReCons_Prot(out_t, 1000, 32, 7)

        Ipg, att_maps_pg = self.bcn(out_t, x)
        Ipc, att_maps_pc = self.bcn(out_t, conv_xd)
        Ipt, att_maps_pt = self.bcn(out_t, sub_s)

        print("Interation Map:", Ipg.shape, Ipc.shape, Ipt.shape)


        # gcs_xd = self.gcs_attention(torch.cat((x,sub_s,conv_xd), 1))

        xc = torch.cat((Ipg, Ipc, Ipt), 1)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


