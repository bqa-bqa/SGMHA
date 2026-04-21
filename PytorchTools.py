import torch
import torch.nn as nn
import torch.nn.functional as F

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss

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
        Wh = torch.matmul(h, self.W)  # (N, out_channels)
        
        edge_index = (adj > 0).nonzero(as_tuple=False).t()  # (2, E)
        row, col = edge_index 
        
        Wh_i = Wh[row] 
        Wh_j = Wh[col] 
        edge_features = torch.cat([Wh_i, Wh_j], dim=1) 
        
        edge_e = torch.matmul(edge_features, self.a).squeeze()  # (E,)
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