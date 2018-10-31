import numpy as np
from torch import nn
import torch

class EncoderLSTM(nn.Module):

    def __init__(self, idim, cdim, drop):
        super(EncoderLSTM, self).__init__()
        self.idim = idim
        self.odim = cdim
        self.cdim = cdim
        self.controller = nn.LSTM(idim, cdim)
        self.dropout = nn.Dropout(drop)
        self._reset_controller()

        self.h0 = nn.Parameter(torch.randn(cdim) * 0.05, requires_grad=True)
        self.c0 = nn.Parameter(torch.randn(cdim) * 0.05, requires_grad=True)

    def _reset_controller(self):
        for p in self.controller.parameters():
            if p.dim() == 1:
                nn.init.constant(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.idim + self.cdim))
                nn.init.uniform(p, -stdev, stdev)

    def forward(self, **input):
        embs = input['embs']
        embs = self.dropout(embs)
        bsz = embs.shape[1]

        h = self.h0.expand(1, bsz, self.cdim).contiguous()
        c = self.c0.expand(1, bsz, self.cdim).contiguous()

        hs = []
        cs = []
        os = []
        # os, (h, c) = self.controller(embs, (h, c))
        for emb in embs:
            controller_outp, (h, c) = self.controller(emb.unsqueeze(0), (h, c))
            o = self.dropout(controller_outp)

            hs.append(h)
            cs.append(c)
            os.append(o)

        os = torch.cat(os, dim=0)
        cs = torch.cat(cs, dim=0)
        np.savetxt('lstm_cells.txt', cs[:, 0].cpu().numpy())

        return os
