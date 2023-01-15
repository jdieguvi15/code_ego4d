
import torch.nn as nn
import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

class PositionWiseFFN(nn.Module):
    """Red neuronal densa que aplicaremos al final de cada paso para obtener nueva info"""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.af = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(ffn_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X):
        return self.dropout2(self.dense2(self.dropout1(self.af(self.dense1(x)))))

class AddNorm(nn.Module):
    """Suma y normaliza"""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class TransformerEncoderLevel(nn.Module):
    """Transformer encoder Level."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, mask_size, testing=False, use_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.mask_size = mask_size
        self.testing=testing
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        #Hacemos uns máscara para estudiar atención local
        nqk = X.shape[1]
        mask = torch.full((1, self.num_heads, nqk, nqk), float('-inf'))
        i = 0
        while(nqk > 0):
            mask[:,:,i*self.mask_size:(i+1)*self.mask_size,i*self.mask_size:(i+1)*self.mask_size] = 0
            i += 1
            nqk -= self.mask_size
        mask[:,:,i*self.mask_size:,i*self.mask_size:] = 0
        mask = mask.to('cuda')
        
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, window_mask=mask))
        Z = self.addnorm2(Y, self.ffn(Y))
        if self.testing:
            print("Encoder Block: Z=", Z.shape)
        return Z
        
class TransformerEncoder(d2l.Encoder):
    """Transformer encoder."""
    def __init__(self, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, num_levels, dropout, mask_size, use_bias=False, testing=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.testing = testing
        
        #quitamos el pos encoding
        #self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        self.convs = nn.ModuleList()
        for i in range(num_levels):
            self.blks.add_module("Level"+str(i), TransformerEncoderLevel(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, mask_size, testing, use_bias))
            
            if i == 0:
                self.convs.append(nn.Sequential(
                    nn.Conv1d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1, groups=1),
                    nn.ReLU()))
            else:
                self.convs.append(nn.Sequential(
                    #nn.Conv1d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=2, padding=1, groups=1),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.ReLU()))

    def forward(self, X, valid_lens):
        #X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        if self.testing:
            print("Encoder: x0=", X.shape)
        self.attention_weights = [None] * len(self.blks)
        feats = []
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
            feats.append(X)
            if self.testing:
                print("Encoder: X", i, "=", X.shape)
            #Reducimos el tamaño a cada iteración
            X = X.transpose(1, 2)
            X = self.convs[i](X) #no hay una forma mejor?
            X = X.transpose(1, 2)
        if self.testing:
            print("Encoder: feats=", [e.shape for e in feats])
        return feats
        
class TransformerDecoderLevel(nn.Module):
    # The i-th Level in the transformer decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i, num_levels, testing=False):
        super().__init__()
        self.i = i
        self.num_levels = num_levels
        self.testing=testing
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)
        
        if i == num_levels -2:
            self.deconv = nn.Sequential(
                nn.ConvTranspose1d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3,stride=1,padding=1, output_padding=1, groups=1),
                nn.ReLU())
        else:
            self.deconv = nn.Sequential(
                nn.ConvTranspose1d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3,stride=2,padding=1, output_padding=1, groups=1),
                nn.ReLU())

    def forward(self, feats_enc, feats_dec):
        feats_dec = self.deconv(feats_dec.transpose(1,2)).transpose(1,2)
        
        #Mix entre los dos métodos, buscamos relaciones en cada conjunto con respecto del otro y lo concatenamos
        X = self.attention1(feats_enc, feats_dec, feats_dec, None)
        X2 = self.addnorm1(X, feats_enc)
        Y = self.attention2(feats_dec, feats_enc, feats_enc, None)
        Y2 = self.addnorm2(Y, feats_dec)
        
        if self.testing:
            print("Decoder block: feats_enc=", feats_enc.shape)
            print("Decoder block: feats_dec=", feats_dec.shape)
            print("Decoder block: X2=", X2.shape)
            print("Decoder block: Y2=", Y2.shape)
        #Z = torch.cat((X2, Y2), 1) #los concatenamos por los ejes temporales
        Z = X2 + Y2     #Así tiene la misma forma que en el baseline
        return self.addnorm3(Z, self.ffn(Z))
        
class TransformerDecoder(d2l.AttentionDecoder):
    """
    Está basado en el Decoder de los transformers de palabras pero cada iteración se aplica a un output del encoder, no cada "palabra"
    """
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, num_levels, dropout, testing=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.num_levels = num_levels
        self.testing = testing
        
        #Para el primer paso:
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm = AddNorm(num_hiddens, dropout)
        
        self.blks = nn.Sequential()
        for i in range(num_levels - 1):
            self.blks.add_module("Level"+str(i), TransformerDecoderLevel(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i, num_levels, testing))

    def forward(self, input):
        #X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        #estamos tratando con features que ya han pasado un positional encoding, repetimos? creo que no
        
        #La primera iteración la hacemos por separado
        X = input[-1]
        X2 = self.attention(X, X, X, None)
        feats_dec = self.addnorm(X, X2)
        
        #TODO meter la atención para el primer bloque aquí
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        
        feats = [feats_dec]
        for i, blk in enumerate(self.blks):
            ii = self.num_levels - i - 2
            feats_enc = input[ii]
            feats_dec = blk(feats_enc, feats_dec)
            # Decoder attention1 weights
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # Decoder attention2 weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
            feats.append(feats_dec)
            if self.testing:
                print("Decoder: feats_dec", i, "=", feats_dec.shape)
        return feats

    @property
    def attention_weights(self):
        return self._attention_weights

class Transformer2Tests(nn.Module):
    """La clase base para construir el transformer"""
    def __init__(self, opt):
        super(Transformer2Tests, self).__init__()
        self.input_feat_dim = opt["input_feat_dim"]
        self.bb_hidden_dim = opt['bb_hidden_dim']
        self.tem_best_loss = 10000000
        self.num_levels = opt['num_levels']
        self.testing = opt['testing']
        self.mask_size = opt['mask_size']

        #Reducimos el espacio de Features de 2304 a 256 - TODO buscar otro método mejor
        self.emb = nn.Sequential(
            nn.Conv1d(in_channels=self.input_feat_dim, out_channels=self.bb_hidden_dim, kernel_size=3,stride=1,padding=1,groups=1),
            nn.ReLU(),
        )
        
        #PARÁMETROS:
        #num_hiddens es la dimensión con la que representaremos los datos, en este caso es la largada del vector de features (temporal steps, se irá reduciendo)
        num_hiddens = self.bb_hidden_dim   #numero de elementos de cada feature
        num_blks = opt['num_blks']    #de momento solo 1 blk
        dropout = 0.2   #lo recomendado en el libro
        ffn_num_hiddens = opt["mlp_num_hiddens"]    #es lo mismo
        num_heads = opt["num_heads"]
        
        #el parámetro tgt_pad se ha eliminado porque se usa para la loss y este modelo no llega a classificar, solo Data Augmentation
        
        
        self.encoder = TransformerEncoder(num_hiddens, ffn_num_hiddens, num_heads, num_blks, self.num_levels, dropout, self.mask_size, testing=self.testing)
        self.decoder = TransformerDecoder(num_hiddens, ffn_num_hiddens, num_heads, num_blks, self.num_levels, dropout, testing=self.testing)
        
    def forward(self, input, *args):
    
        if self.testing:
            print("Transformer: input.shape:", input.shape)
    
        X = self.emb(input)
        X = X.transpose(1, 2)
    
        feats_enc = self.encoder(X, None, *args)
        feats_dec = self.decoder(feats_enc)
        
        return feats_enc, feats_dec
