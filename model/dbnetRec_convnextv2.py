# dbnet_convnextv2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# import your convnextv2 builders
# from convnextv2 import convnextv2_small, convnextv2_nano, convnext_medium, convnextv2_large



class ConvNeXtV2_BiLSTM_CTC(nn.Module):
    def __init__(self, backbone, in_channels_c5: int, vocab_size: int,
                 d_model: int = 256, hidden: int = 256, num_layers: int = 2):
        super().__init__()
        self.backbone = backbone

        # C5 -> d_model
        self.proj = nn.Conv2d(in_channels_c5, d_model, kernel_size=1)

        # sequence model (width as time)
        self.rnn = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=num_layers,
            bidirectional=True
        )

        self.classifier = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        feats = self.backbone(x, return_feats=True)
        c5 = feats[-1]                      # (B, C5, H', W')

        f = self.proj(c5)                   # (B, d_model, H', W')
        f = f.mean(dim=2)                   # avg over height -> (B, d_model, W')
        f = f.permute(2, 0, 1).contiguous() # (T=W', B, d_model)

        f, _ = self.rnn(f)                  # (T, B, 2*hidden)
        logits = self.classifier(f)         # (T, B, V)

        lengths = torch.full((logits.shape[1],), logits.shape[0],
                             dtype=torch.long, device=logits.device)
        return logits, lengths

