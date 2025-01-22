import torch
import torch.nn as nn
from src.vq_vae_pytorch.vq import Bottleneck
from src.vq_vae_pytorch.jukebox import Encoder, Decoder


class Quantizer(nn.Module):
    def __init__(self, h):
        super().__init__()

        self.encoder = Encoder(**h.f0_encoder_params)
        self.vq = Bottleneck(**h.f0_vq_params)
        self.decoder = Decoder(**h.f0_decoder_params)

    def forward(self, f0):
        f0_h = self.encoder(f0)
        _, f0_h_q, f0_commit_losses, f0_metrics = self.vq(f0_h)
        f0 = self.decoder(f0_h_q)

        return f0, f0_commit_losses, f0_metrics
    
    def get_code(self, f0):
        f0_h = self.encoder(f0)
        code, _, _, _ = self.vq(f0_h)
        return code[0]


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2