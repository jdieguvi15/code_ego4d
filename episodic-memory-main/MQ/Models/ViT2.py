# codigo separado en funciones

import numpy as np
import json
import os.path
import torch
from torch import nn
from d2l import torch as d2l
import torchvision
from torchvision import transforms
import collections
import inspect

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiHeadAttention2(d2l.MultiHeadAttention):
    """Modificación en el Multi-head attention para permitir proyectar la atención en un espacio más grande"""
    def __init__(self, dim_attention, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__(num_hiddens, num_heads, dropout, bias, **kwargs)
        #self.num_heads = num_heads
        #self.attention = d2l.DotProductAttention(dropout, num_heads)
        self.W_q = nn.LazyLinear(dim_attention, bias=bias)
        self.W_k = nn.LazyLinear(dim_attention, bias=bias)
        self.W_v = nn.LazyLinear(dim_attention, bias=bias)
        #self.W_o = nn.LazyLinear(num_hiddens, bias=bias) #se definen en el super
        if(dim_attention % num_heads != 0): print("Error: dim_attention no es divisible por num_heads")
    
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
    def __init__(self, dim_attention, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = MultiHeadAttention2(dim_attention, num_hiddens, num_heads, dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout) #alomejor esto es redundante si solo hay 1 bloque

    def forward(self, X, valid_lens=None):
        X = self.ln1(X)                                         # se puede probar a quitar
        return X + self.mlp(self.ln2(
            X + self.attention(X, X, X, valid_lens)))
    
class ViT2(nn.Module):
    """Vision transformer.
    Modificación para que solo aplique el positional embedding, bloques de encoder y head para dar el formato de salida.
    He tocado lo mínimo del Vit original, vamos a ver como va.
    
    (mlp_num_hiddens dice las neuronas que tendrán los mlp en la capa intermedia)
    (num_hiddens dice cuantos valores tiene cada feature) -> podria modificarlo para calcular con más valores
    Además, por como está codificado el código del libro, num_heads ha de ser divisor de num_hiddens -> demasiado rígido
    Problema: divisores de 58: 2 y 29
    (num_hiddens = largada del vector features de momento, lo más simple)
    
    input: (Cin, Lin) = (num_features, num_hiddens)
    output: (Cout, Lin) = (out_channels, num_hiddens / 2)   num_features = out_channels
    
    
    
    
    !!!!!! POSIBLE PROBLEMA: que se ejecute una a una y no pueda ver las relaciones entre los features si solo ve 1 a la vez
    """
    def __init__(self, num_features, num_temp, mlp_num_hiddens, dim_attention, num_heads, stride, num_blks=1, emb_dropout=0.1, blk_dropout=0.1, use_bias=False, usewandb=False, testing=False):
        super().__init__()
        self.save_hyperparameters()
        print("---- Creamos un ViT 2.0 ----")
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_temp, num_features))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                dim_attention, num_features, num_features, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        #self.head = nn.Sequential(nn.LayerNorm(num_temp), nn.Linear(num_temp, num_hiddens_out))
                                  
        self.head = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=stride, padding=1, groups=1),
            nn.ReLU(inplace=True))
                                  
    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
    
        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

    def forward(self, X):
        if self.testing:
            print("ViT: X.shape:", X.shape)
        X = X.transpose(1, 2)
        if self.testing:
            print("ViT: X.shape transposed:", X.shape)
        #Positional embedding
        X = self.dropout(X + self.pos_embedding)
        #Bloques de atención
        for blk in self.blks:
            X = blk(X)
        if self.testing:
            print("ViT: X.shape_after:", X.shape)
        X = X.transpose(1, 2)
        #Reducimos el tamaño para la siguiente iteración
        X = self.head(X)
        if self.testing:
            print("ViT: X.shape_after_head:", X.shape)
        return X
        
    
class PatchEmbedding(nn.Module):
    def __init__(self, vect_size=928, patch_size=8, num_hiddens=768):
        super().__init__()
        self.num_patches = vect_size // patch_size
        self.conv = nn.LazyConv1d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        if self.testing:
            print("X.shape: ", X.shape)
        Y = self.conv(X)
        if self.testing:
            print("Y.shape: ", Y.shape)
        Z = Y.transpose(1, 2)
        if self.testing:
            print("Z.shape: ", Z.shape)
        return Z
