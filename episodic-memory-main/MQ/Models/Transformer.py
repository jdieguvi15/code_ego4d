
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
        self.relu = nn.ReLU()
        #self.gelu = nn.GELU()
        #self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(ffn_num_outputs)
        #self.dropout2 = nn.Dropout(dropout)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
        #return self.dropout2(self.dense2(self.dropout1(self.gelu(self.dense1(x)))))

class AddNorm(nn.Module):
    """Suma y normaliza"""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
        
class TransformerEncoder(d2l.Encoder):
    """Transformer encoder."""
    def __init__(self, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False, vocab_size=0):
        super().__init__()
        self.num_hiddens = num_hiddens
        #embeeding es como un feature extracton pero ya trabajamos con features
        #self.embedding = nn.Embedding(vocab_size, num_hiddens) #creo que no hace falta
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        #X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        if self.testing:
            print("Encoder: x0=", x.shape)
        X = self.pos_encoding(X)
        if self.testing:
            print("Encoder: x0 with pos encoding=", x.shape)
        self.attention_weights = [None] * len(self.blks)
        feats = []
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
            feats.append(x)
            if self.testing:
                print("Encoder: x", i, "=", x.shape)
            #TODO reducir el tamaño a cada iteración
        if self.testing:
            print("Encoder: feats=", [e.shape for e in feats])
        return feats
        
class TransformerDecoderBlock(nn.Module):
    # The i-th block in the transformer decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so state[2][self.i] is None as initialized. When
        # decoding any output sequence token by token during prediction,
        # state[2][self.i] contains representations of the decoded output at
        # the i-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # Shape of dec_valid_lens: (batch_size, num_steps), where every
            # row is [1, 2, ..., num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention. Shape of enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
        
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, vocab_size=0):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        #self.embedding = nn.Embedding(vocab_size, num_hiddens) #creo que no hace falta
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.LazyLinear(vocab_size) #TODO!!!

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        #X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        X = self.pos_encoding(X)
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
            if self.testing:
                print("Decoder: x", i, "=", x.shape)
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

class Transformer(nn.Module):
    """La clase base para construir el transformer"""
    def __init__(self, opt):
        super(Transformer, self).__init__()
        self.input_feat_dim = opt["input_feat_dim"]
        self.bb_hidden_dim = opt['bb_hidden_dim']
        self.tem_best_loss = 10000000
        self.num_levels = opt['num_levels']
        self.testing = opt['testing']

        #Reducimos el espacio de Features de 2304 a 256 - TODO buscar otro método mejor
        self.conv0 = nn.Sequential(
            nn.Conv1d(in_channels=self.input_feat_dim, out_channels=self.bb_hidden_dim,kernel_size=3,stride=1,padding=1,groups=1),
            nn.ReLU(inplace=True),
        )
        
        #PARÁMETROS:
        #num_hiddens es la dimensión con la que representaremos los datos, en este caso es la largada del vector de features (temporal steps, se irá reduciendo)
        num_hiddens = 928   #máximo de frames que puede tener el vídeo
        num_blks = opt['num_levels']    #cada bloque hará la función de un level
        dropout = 0.2   #lo recomendado en el libro
        ffn_num_hiddens = opt["mlp_num_hiddens"]    #es lo mismo
        num_heads = opt["num_heads"]
        
        #el parámetro tgt_pad se ha eliminado porque se usa para la loss y este modelo no llega a classificar, solo Data Augmentation
        
        
        self.encoder = TransformerEncoder(num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout)
        self.decoder = TransformerDecoder(num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout)
        
    def forward(self, input, *args):
    
        if self.testing:
            print("Transformer: input.shape:", input.shape)
    
        X = self.conv0(input)
        X = X.transpose(1, 2)
        if self.testing:
            print("ViT: X.shape ready:", X.shape)
    
        feats_enc = self.encoder(input, None, *args)
        
        dec_state = self.decoder.init_state(feats_enc,*args)
        feats_dec = self.decoder(feats_enc, dec_state)[0]
        
        return feats_enc, feats_dec

"""
Falta por hacer:
- OJO que el encoder da como output un monton de features y el decoder requiere otr input creo, cuadrar eso
- entender como funcionan los states del decoder y qué está devolviendo
- hacer que el encoder y el decoder vayan append los feats de cada iteración
- Comprovar parecidos con ViT2 y con xGPN y comprovar que se hacen todos los prints
- Ajustar que el input que recibe el transformer y el output que da sean correctos
- Hacer que se reduzca la dimensión para cada bloque (o no)
- Revisar los comentarios de todo y poner comentarios explicativos
"""
