# codigo separado en funciones

import numpy as np
import json
import os.path
import torch
from torch import nn
from d2l import torch as d2l
import torchvision
from torchvision import transforms

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))
    
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout) #alomejor esto es redundante si solo hay 1 bloque

    def forward(self, X, valid_lens=None):
        X = self.ln1(X)
        return X + self.mlp(self.ln2(
            X + self.attention(X, X, X, valid_lens)))
    
class ViT(d2l.Classifier):
    """Vision transformer.
    Modificación para que solo aplique el positional embedding, bloques de encoder y head para dar el formato de salida.
    He tocado lo mínimo del Vit original, vamos a ver como va.
    
    (mlp_num_hiddens dice las neuronas que tendrán los mlp en la capa intermedia)
    (num_hiddens dice cuantos valores tiene cada feature)
    *podria modificarlo ?
    
    input: (Cin, Lin) = (in_channels, num_hiddens)
    output: (Cout, Lin) = (out_channels, num_hiddens / 2)   in_channels = out_channels
    """
    def __init__(self, in_channels, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks=1, emb_dropout=0.1, blk_dropout=0.1,
                 use_bias=False, usewandb=False):
        super().__init__()
        self.save_hyperparameters()
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, in_channels, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_hiddens//2))

    def forward(self, X):
        #No hace patch embedding ni añade un token para la clase
        #Hacemos positional embedding y el vit block
        #Al final le damos el formato del output con linear
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X)
    
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        #self.plot('loss', l, train=True)
        #a = self.accuracy(self(*batch[:-1]), batch[-1])
        #wandb.log({"train_loss": l, "train_acc": a})
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        #a = self.accuracy(self(*batch[:-1]), batch[-1])
        #wandb.log({"val_loss": l, "val_acc": a})
        #self.plot('loss', l, train=False)
        
    #def configure_optimizers(self):
    #    return eval("torch.optim."+ self.optimizer_name + "(self.parameters(), lr=self.lr)")
    

