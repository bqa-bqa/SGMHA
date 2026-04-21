import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from PytorchTools import GATConv, GCNConv, sce_loss

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
        
        self.criterion = self.setup_loss_fn(loss_fn='sce')

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
        
        train_tf = tfa_embed[train_sample[:, 0]]
        train_target = targeta_embed[train_sample[:, 1]]

        pred = torch.mul(train_tf, train_target)
        pred = torch.sum(pred, dim=1).view(-1, 1)

        return pred