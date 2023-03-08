
import torch.nn as nn
import math
import pandas as pd
import torch
from d2l import torch as d2l

class MultiHeadAttention2(d2l.MultiHeadAttention):
    """Modification in Multi-head attention to allow attention to be projected in a larger space"""
    def __init__(self, dim_attention, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__(num_hiddens, num_heads, dropout, bias, **kwargs)
        self.W_q = nn.LazyLinear(dim_attention, bias=bias)
        self.W_k = nn.LazyLinear(dim_attention, bias=bias)
        self.W_v = nn.LazyLinear(dim_attention, bias=bias)
        if(dim_attention % num_heads != 0):
            print("Error: dim_attention no es divisible por num_heads")

class PositionWiseFFN(nn.Module):
    """Dense neural network that we will apply at the end of each step to
     obtain new information"""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.af = nn.GELU()
        self.dropout1 = nn.Dropout(0.1)
        self.dense2 = nn.LazyLinear(ffn_num_outputs)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, X):
        return self.dropout2(self.dense2(self.dropout1(self.af(self.dense1(X)))))

class AddNorm(nn.Module):
    """Residual layer"""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class TransformerEncoderLevel(nn.Module):
    # The i-th Level in the transformer encoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, dim_attention, mask_size, testing=False, use_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.mask_size = mask_size
        self.testing=testing
        self.dim_attention = dim_attention
        self.attention = MultiHeadAttention2(dim_attention, num_hiddens, num_heads,
                                                dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        # build a mask for local sef attention
        nqk = X.shape[1]
        mask = torch.full((1, self.num_heads, nqk, nqk), float('-inf'))
        i = 0
        while(nqk > self.mask_size):
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
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, num_blks, num_levels,
                dropout, dim_attention, mask_size, use_bias=False, testing=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.testing = testing
        
        self.blks = nn.Sequential()
        self.convs = nn.ModuleList()
        for i in range(num_levels):
            self.blks.add_module("Level"+str(i), TransformerEncoderLevel(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, dim_attention, mask_size, testing, use_bias))
            
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels=num_hiddens, out_channels=num_hiddens,
                          kernel_size=3, stride=2, padding=1, groups=1),
                #nn.MaxPool1d(kernel_size=2, stride=2),
                nn.GELU()))

    def forward(self, X, valid_lens):
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

            X = X.transpose(1, 2)
            X = self.convs[i](X) # reduction of the temporal dimension
            X = X.transpose(1, 2)
        if self.testing:
            print("Encoder: feats=", [e.shape for e in feats])
        return feats
        
class TransformerDecoderLevel(nn.Module):
    # The i-th Level in the transformer decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, dim_attention,
                 i, num_levels, testing=False):
        super().__init__()
        self.i = i
        self.num_levels = num_levels
        self.testing=testing
        self.attention1 = MultiHeadAttention2(dim_attention, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = MultiHeadAttention2(dim_attention, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)
        self.dim_attention = dim_attention
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=num_hiddens, out_channels=num_hiddens,
                            kernel_size=3,stride=2,padding=1, output_padding=1, groups=1),
            nn.GELU())

    def forward(self, feats_enc, feats_dec):
        feats_dec = self.deconv(feats_dec.transpose(1,2)).transpose(1,2)
        
        # Attention to both sides
        X = self.attention1(feats_enc, feats_dec, feats_dec, None)
        X2 = self.addnorm1(X, feats_enc)
        Y = self.attention2(feats_dec, feats_enc, feats_enc, None)
        Y2 = self.addnorm2(Y, feats_dec)
        
        #Alternative method: doesn't require convolution and allows for more levels
        #X2 = self.attention1(feats_enc, feats_dec, feats_dec, None)
        #Y2 = feats_enc
        
        if self.testing:
            print("Decoder block: feats_enc=", feats_enc.shape)
            print("Decoder block: feats_dec=", feats_dec.shape)
            print("Decoder block: X2=", X2.shape)
            print("Decoder block: Y2=", Y2.shape)
        #Z = torch.cat((X2, Y2), 1) #they can be concatenated - not recomended
        Z = X2 + Y2
        return self.addnorm3(Z, self.ffn(Z))
        
class TransformerDecoder(d2l.AttentionDecoder):
    """
    It is based on the Transformers Decoder for text but each iteration is applied to an
     output of the encoder, not an output of the word transformer.
    """
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, num_levels, dropout, dim_attention, testing=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.num_levels = num_levels
        self.testing = testing
        self.dim_attention = dim_attention
        
        self.attention = MultiHeadAttention2(dim_attention, num_hiddens, num_heads, dropout)
        self.addnorm = AddNorm(num_hiddens, dropout)
        
        self.blks = nn.Sequential()
        for i in range(num_levels - 1):
            self.blks.add_module("Level"+str(i), TransformerDecoderLevel(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, dim_attention, i, num_levels, testing))

    def forward(self, input):
        # The first iteration is done separately
        X = input[-1]
        X2 = self.attention(X, X, X, None)
        feats_dec = self.addnorm(X, X2)
        
        # We save the attention values
        self._attention_weights = [self.attention.attention.attention_weights]
        self._attention_weights.extend([[None] * len(self.blks) for _ in range (2)])
        
        feats = [feats_dec]
        for i, blk in enumerate(self.blks):
            ii = self.num_levels - i - 2
            feats_enc = input[ii]
            feats_dec = blk(feats_enc, feats_dec)
            # Decoder attention1 weights
            self._attention_weights[1][i] = blk.attention1.attention.attention_weights
            # Decoder attention2 weights
            self._attention_weights[2][i] = blk.attention2.attention.attention_weights
            feats.append(feats_dec)
            if self.testing:
                print("Decoder: feats_dec", i, "=", feats_dec.shape)
        return feats

    @property
    def attention_weights(self):
        return self._attention_weights

class ReMoT(nn.Module):
    """The base class to construct the Transformer"""
    def __init__(self, opt):
        super(ReMoT, self).__init__()
        print("Creating the Transformer Pyramid")
        self.bb_hidden_dim = opt['bb_hidden_dim']
        self.tem_best_loss = 10000000
        self.num_levels = opt['num_levels']
        self.testing = opt['testing']
        self.mask_size = opt['mask_size']
        self.mlp_num_hiddens = opt['mlp_num_hiddens']
        self.dim_attention = opt['dim_attention']
        
        self.features = opt['features']
        self.input_feat_dim = 0
        if 's' in self.features:
            self.input_feat_dim += opt['slowfast_dim']
        if 'o' in self.features:
            self.input_feat_dim += opt['omnivore_dim']
        if 'e' in self.features:
            self.input_feat_dim += opt['egovlp_dim']

        # Reduce the feature dimension from input_feat_dim to bb_hidden_dim
        self.embC = nn.Sequential(
            nn.Conv1d(in_channels=self.input_feat_dim, out_channels=self.bb_hidden_dim, kernel_size=3,stride=1,padding=1,groups=1),
            nn.GELU(),)
        # second option:
        #self.emb = nn.Sequential(
        #    nn.LazyLinear(self.mlp_num_hiddens),
        #    nn.GELU(),
        #    nn.LazyLinear(self.bb_hidden_dim),
        #    nn.GELU(),)

        # PARAMETERS:
        # num_hiddens is the dimension with which we represent the data,
        # in this case, the temporal steps, it will be reduced
        num_hiddens = self.bb_hidden_dim   # number of elements in each feature
        num_blks = opt['num_blks']    # we will only use 1
        self.dropout = 0.2
        ffn_num_hiddens = opt["mlp_num_hiddens"]    # it's basically the same
        num_heads = opt["num_heads"]
        
        #Pos enc can be removed
        #self.pos_encoding = nn.Parameter(torch.randn(1, 928, num_hiddens)) # learnable
        #self.pos_encoding = d2l.PositionalEncoding(num_hiddens, 0.1) # fixed
        
        self.encoder = TransformerEncoder(num_hiddens, ffn_num_hiddens, num_heads, num_blks, self.num_levels, self.dropout, self.dim_attention, self.mask_size, testing=self.testing)
        self.decoder = TransformerDecoder(num_hiddens, ffn_num_hiddens, num_heads, num_blks, self.num_levels, self.dropout, self.dim_attention, testing=self.testing)
        
    def forward(self, input, *args):
    
        if self.testing:
            print("Transformer: input.shape:", input.shape)
    
        X = self.embC(input)
        X = X.transpose(1, 2)
        #X = self.emb(input)
        #X = X + self.pos_encoding
    
        feats_enc = self.encoder(X, None, *args)
        feats_dec = self.decoder(feats_enc)
        
        return feats_enc, feats_dec
