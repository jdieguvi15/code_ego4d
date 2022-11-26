
import torch.nn as nn
# from Utils.Sync_batchnorm.batchnorm import SynchronizedBatchNorm1d
from .GCNs import xGN
from .ViT import ViT



class XGPN(nn.Module):
    def __init__(self, opt):
        super(XGPN, self).__init__()
        self.input_feat_dim = opt["input_feat_dim"]
        self.bb_hidden_dim = opt['bb_hidden_dim']
        self.batch_size = opt["batch_size"]
        self.tem_best_loss = 10000000
        self.num_levels = opt['num_levels']
        self.use_ViT = opt['use_ViT']
        self.use_xGPN = opt['use_xGPN']

        self.conv0 = nn.Sequential(
            nn.Conv1d(in_channels=self.input_feat_dim, out_channels=self.bb_hidden_dim,kernel_size=3,stride=1,padding=1,groups=1),
            nn.ReLU(inplace=True),
        )

        self.levels_enc = nn.ModuleList()
        num_hiddens = 928 # viene determinado por la forma de los features de SlowFast
        for i in range(self.num_levels):
            if i == 0:
                stride = 1
                num_hiddens = 928
            else:
                stride = 2
                num_hiddens = num_hiddens // 2
            # Añado num_hiddens para controlar como decrece
            self.levels_enc.append(self._make_levels_enc(opt, in_channels=self.bb_hidden_dim, out_channels=self.bb_hidden_dim, num_hiddens=num_hiddens, stride = stride))

        self.levels_dec = nn.ModuleList()
        for i in range(self.num_levels - 1):
            output_padding = 1
            self.levels_dec.append(self._make_levels_dec(in_channels=self.bb_hidden_dim, out_channels=self.bb_hidden_dim, output_padding = output_padding))

        self.levels1 = nn.ModuleList()
        for i in range(self.num_levels):
            self.levels1.append(self._make_levels(in_channels=self.bb_hidden_dim, out_channels=self.bb_hidden_dim))

        self.levels2 = nn.ModuleList()
        for i in range(self.num_levels - 1):
            self.levels2.append(self._make_levels(in_channels=self.bb_hidden_dim, out_channels=self.bb_hidden_dim))


    def _make_levels_enc(self, opt, in_channels, out_channels, num_hiddens, stride = 2):
        if self.use_ViT:
            print("---- Creamos un ViT ----")
            # in_channels, num_hiddens, mlp_num_hiddens, num_heads
            return ViT(in_channels, num_hiddens, 2048, 8)
        elif self.use_xGPN:
            return xGN(opt, in_channels=in_channels, out_channels=out_channels, stride = stride)
        else:
            #input: (Cin, Lin), output: (Cout, Lout), en aquest cas Cin = Cout será el numero de frames?
            #la segunda dimensión si que se va reduciendo, empieza en 928 y se va reduciendo a la mitad en cada level hasta 58
            return nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                          groups=1),
                nn.ReLU(inplace=True)
            )

    def _make_levels_dec(self, in_channels, out_channels, output_padding = 1):
        #esto si funciona con xGN deberia funcional igual con ViT pero parece un poco sorprendente
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,kernel_size=3,stride=2,padding=1, output_padding=output_padding, groups=1),
            nn.ReLU(inplace=True),
        )

    def _make_levels(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=3,stride=1,padding=1,groups=1),
            nn.ReLU(inplace=True),
        )

    def _encoder(self, input, num_frms):

        feats = []
        x = self.conv0(input)
        print("x0:", x.shape)
        for i in range(0, self.num_levels):
            if self.use_xGPN:
                x = self.levels_enc[i](x, num_frms)
            else:
                x = self.levels_enc[i](x)
            feats.append(x)
            print("x", i, ":", x.shape)

        return feats

    def _decoder(self, input):

        feats = []
        x = self.levels1[0](input[self.num_levels - 1])
        feats.append(x)

        for i in range(self.num_levels - 1):
            ii = self.num_levels - i - 2
            feat_enc = self.levels2[i](input[ii])
            feat_dec = self.levels_dec[i](x)
            x = self.levels1[i+1](feat_enc + feat_dec)
            feats.append(x)



        return feats

    def forward(self, input, num_frms):

        feats_enc = self._encoder(input, num_frms)
        feats_dec = self._decoder(feats_enc)

        return feats_enc, feats_dec

