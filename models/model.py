# models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
from data.data import AtomFeatureEncoder

class GCN_Layer(torch.nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len, num_nbr):
        super(GCN_Layer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.num_nbr = num_nbr
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, ):
        atom_nbr_fea = x[edge_index[1], :]
        atom_init_fea = x[edge_index[0], :]
        total_nbr_fea = torch.cat((atom_nbr_fea, atom_init_fea, edge_attr), dim=1)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea)

        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=1)
        nbr_filter = torch.sigmoid(nbr_filter)
        nbr_core = F.softplus(nbr_core)

        nbr_sumed = nbr_core * nbr_filter
        # nbr_sumed.shape=(N * M, atom_fea_len)
        nbr_sumed = nbr_sumed.reshape((-1, self.num_nbr, self.atom_fea_len))
        nbr_sumed = torch.sum(nbr_sumed, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = F.softplus(x + nbr_sumed)
        return out

class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        full_atom_feature_dims = AtomFeatureEncoder.get_full_dims()
        self.atom_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])
        return x_embedding

class Net(torch.nn.Module):

    def __init__(self, nbr_fea_len=41, atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False, n_classes=2, attention=False, dynamic_attention=False, n_heads=1,
                 max_num_nbr=12, pooling='mean', p=0):
        super(Net, self).__init__()
        self.classification = classification
        self.pooling = pooling
        self.bn = nn.BatchNorm1d(atom_fea_len)
        self.embedding = AtomEncoder(atom_fea_len)
        #self.embedding = nn.Embedding(101, atom_fea_len)
        self.p = p
        if attention:
            self.n_convs = nn.ModuleList([torch_geometric.nn.GATConv(atom_fea_len, atom_fea_len,
                                                                     edge_dim=nbr_fea_len, heads=n_heads, concat=False)
                                          for _ in range(n_conv)])
        elif dynamic_attention:
            self.n_convs = nn.ModuleList([torch_geometric.nn.GATv2Conv(atom_fea_len, atom_fea_len,
                                                                       edge_dim=nbr_fea_len, heads=n_heads, concat=False)
                                          for _ in range(n_conv)])
        else:
            self.n_convs = nn.ModuleList([GCN_Layer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len, num_nbr=max_num_nbr)
                                          for _ in range(n_conv)])

        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h - 1)])
        if self.p > 0:
            self.dropout = nn.Dropout(p=p)
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, n_classes)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, data):
        x, edge_index, edge_weight, y = data.x, data.edge_index, data.edge_attr, data.y
        batch = data.batch
        x = self.embedding(x)
        # x = x.squeeze(1)

        for conv in self.n_convs:
            x = conv(x=x, edge_index=edge_index, edge_attr=edge_weight)
        #global_pool function
        if self.pooling == 'add':
            x = torch_geometric.nn.global_add_pool(x, batch).unsqueeze(1).squeeze()
        elif self.pooling == 'max':
            x = torch_geometric.nn.global_max_pool(x, batch).unsqueeze(1).squeeze()
        else:
            x = torch_geometric.nn.global_mean_pool(x, batch).unsqueeze(1).squeeze()
        x = F.softplus(x)
        x = self.bn(x)
        x = self.conv_to_fc(x)
        x = F.softplus(x)
        if hasattr(self, 'fcs'):
            for hidden in self.fcs:
                x = F.softplus(hidden(x))
        if self.p > 0:
            x = self.dropout(x)
        out = self.fc_out(x)
        return out
