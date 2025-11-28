# scGNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sce_loss
from functools import partial

class GATConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, alpha, concat=True):
        super(GATConv, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_channels, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        
        edge_index = (adj > 0).nonzero(as_tuple=False).t()
        row, col = edge_index 
        
        Wh_i = Wh[row] 
        Wh_j = Wh[col] 
        edge_features = torch.cat([Wh_i, Wh_j], dim=1)
        
        edge_e = torch.matmul(edge_features, self.a).squeeze()
        edge_e = F.leaky_relu(edge_e, negative_slope=self.alpha)
        
        if self.training:
            keep_mask = torch.rand_like(edge_e) > self.dropout.p
            edge_e = torch.where(
                keep_mask,
                edge_e / (1 - self.dropout.p),
                torch.full_like(edge_e, -1e18)
            )
        
        max_per_target = torch.zeros(Wh.size(0), device=edge_e.device)
        max_per_target.scatter_reduce_(0, col, edge_e, reduce='amax', include_self=False)
        
        exp_edge_e = torch.exp(edge_e - max_per_target[col])
        
        sum_per_target = torch.zeros(Wh.size(0), device=edge_e.device)
        sum_per_target.scatter_add_(0, col, exp_edge_e)
        
        attention_weights = exp_edge_e / (sum_per_target[col] + 1e-16)
        
        h_prime = torch.zeros_like(Wh)
        row = row.to(torch.long)
        col = col.to(torch.long)
        h_prime.scatter_add_(0, col.unsqueeze(1).expand(-1, Wh.size(1)), 
                            Wh_i * attention_weights.unsqueeze(1))

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GCNConv(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.2, bias=True):
        super(GCNConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
    
    def norm_adj(self, adj):
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
        
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        
        adj_normalized = torch.matmul(torch.matmul(D_inv_sqrt, adj), D_inv_sqrt)
        
        return adj_normalized
            
    def forward(self, x, adj):
        adj = self.norm_adj(adj)
        h = torch.matmul(x, self.weight)
        h = torch.matmul(adj, h)
        
        if self.bias is not None:
            h = h + self.bias
        
        return F.relu(h)

class GraphMAE(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, num_hidden, device, batchnorm=True):
        super(GraphMAE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = GATConv(input_dim, num_hidden, dropout=0.6, alpha=0.2)
        self.decoder = GATConv(num_hidden, input_dim, dropout=0.6, alpha=0.2)
        self.encoder_to_decoder = nn.Linear(num_hidden, num_hidden, bias=False)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        
        self.tf_linear = nn.Linear(num_hidden, output_dim)
        self.target_linear = nn.Linear(num_hidden, output_dim)
        self.MLP = nn.Linear(2 * output_dim, 2)
        
        self.criterion = self.setup_loss_fn()

    def encoding_mask_noise(self, x, adj, mask_rate=0.25):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        out_x[mask_nodes] = 0.0
        out_x[mask_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)

    def encode(self, x, adj):
        return self.encoder(x, adj)

    def decode(self, x, adj):
        return self.decoder(x, adj)

    def setup_loss_fn(self, loss_fn='mse'):
        if loss_fn == 'mse':
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=1)
        else:
            raise NotImplementedError
        return criterion

    def _attr_prediction(self, x, adj):
        u_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, adj)
        enc_rep = self.encode(u_x, adj)
        rep = self.encoder_to_decoder(enc_rep)
        rep[mask_nodes] = 0.0
        recon = self.decode(rep, adj)

        x_t = x[mask_nodes]
        x_p = recon[mask_nodes]
        loss = self.criterion(x_t, x_p)

        return loss

    def forward(self, x, adj):
        loss = self._attr_prediction(x, adj)
        loss_item = {'loss': loss.item()}
        return loss, loss_item

    def get_embed(self, x, adj):
        return self.decode(x, adj)

class LinkModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(LinkModel, self).__init__()
        self.tf_linear = nn.Linear(input_dim, hidden_dim)
        self.target_linear = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
        self.tfa_linear = nn.Linear(hidden_dim, output_dim)
        self.targeta_linear = nn.Linear(hidden_dim, output_dim)
        self.gcn = GCNConv(input_dim, hidden_dim)
        
    def forward(self, x, train_sample, adj):
        x_g = self.gcn(x, adj)
        x_g = torch.sigmoid(x_g)
        
        tf_embed = self.tf_linear(x)
        tf_embed = F.leaky_relu(tf_embed)
        
        target_embed = self.target_linear(x)
        target_embed = F.leaky_relu(target_embed)
        
        x_f, _ = self.attention(x_g, tf_embed, target_embed)
        
        tfa_embed = self.tfa_linear(x_f)
        tfa_embed = F.leaky_relu(tfa_embed)
        
        targeta_embed = self.targeta_linear(x_f)
        targeta_embed = F.leaky_relu(targeta_embed)
        
        train_tf = tfa_embed[train_sample[:,0]]
        train_target = targeta_embed[train_sample[:, 1]]

        pred = torch.mul(train_tf, train_target)
        pred = torch.sum(pred, dim=1).view(-1,1)

        return pred